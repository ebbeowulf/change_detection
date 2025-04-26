#!/usr/bin/env python3
import rospy
import cv2
import pdb
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import numpy as np
from threading import Lock
import tf
import message_filters
from camera_params import camera_params
from map_utils_hybrid import pcloud_from_images, pointcloud_open3d
from std_srvs.srv import Trigger, TriggerResponse
from two_query_localize_hybrid import create_object_clusters, calculate_iou, estimate_likelihood
from msg_and_srv.srv import DynamicCluster, DynamicClusterRequest, DynamicClusterResponse, SetInt, SetIntRequest, SetIntResponse
from msg_and_srv.srv import SetString, SetStringResponse, SetStringRequest, SetFloat, SetFloatResponse, SetFloatRequest
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import torch
import datetime
from std_msgs.msg import String
from pcloud_models.srv import EvaluateDetection, EvaluateDetectionResponse
from pcloud_models.msg import DetectionEvaluation, DetectionMetrics
import pickle
import os

TRACK_COLOR=True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalizeAngle(angle):
    while angle>=np.pi:
        angle-=2*np.pi
    while angle<-np.pi:
        angle+=2*np.pi
    return angle

class hybrid_query_localize:
    def __init__(self, query_list, cluster_min_points, detection_threshold, travel_params, storage_dir=None):
        # Initialization of the node, name_sub
        rospy.init_node('hybrid_query_localize', anonymous=True)
        self.listener = tf.TransformListener()
        self.storage_dir = storage_dir

        # Initialize the CvBridge class
        self.bridge = CvBridge()
        self.im_count=1

        self.pose_queue=[]
        self.pose_lock=Lock()
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        
        # Last image stats
        self.last_image=None
        self.travel_params=travel_params

        # Previously detected clusters
        self.known_objects=[]

        # Create Pcloud Data Structures
        self.pcloud_creator=None
        self.query_list=query_list
        self.pcloud=dict()
        for query in self.query_list:
            self.pcloud[query]={'xyz': torch.zeros((0,3),dtype=float,device=DEVICE), 
                               'probs': torch.zeros((0),dtype=float,device=DEVICE), 
                               'rgb': torch.zeros((0,3),dtype=float,device=DEVICE),
                               'clipseg_probs': torch.zeros((0),dtype=float,device=DEVICE),
                               'omdet_probs': torch.zeros((0),dtype=float,device=DEVICE)}
        self.cluster_min_points=cluster_min_points
        self.cluster_iou=0.1
        self.detection_threshold=detection_threshold
        self.cluster_metric='hybrid-mean'
        self.gridcell_size=0.005  # Based on optimized parameters from memory
        
        # Temporal consistency tracking variables
        self.prev_masks = {}  # Store previous masks for each query
        self.prev_probs = {}  # Store previous probability maps for each query
        self.motion_vectors = {}  # Store estimated motion between frames
        self.frame_count = 0  # Counter for frames processed
        self.temporal_weight = 0.3  # Weight for temporal consistency (0-1)
        self.consistency_enabled = True  # Flag to enable/disable temporal consistency
        
        # Evaluation metrics
        self.eval_metrics = {}
        self.ground_truth = {}
        for query in self.query_list:
            self.eval_metrics[query] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_detections': 0,
                'total_points': 0,
                'average_points_per_cluster': 0,
                'point_cloud_density': 0.0
            }
        
        # Setup callback function
        self.camera_params_sub = rospy.Subscriber('/camera_throttled/depth/camera_info', CameraInfo, self.cam_info_callback)
        self.rgb_sub = message_filters.Subscriber('/camera_throttled/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera_throttled/depth/image_rect_raw', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.rgbd_callback)

        # Setup service calls
        self.setClusterMetric_srv = rospy.Service('set_cluster_metric', SetString, self.set_cluster_metric_service)
        self.setclustersize_srv = rospy.Service('set_cluster_size', SetInt, self.set_cluster_size_service)
        self.setiou_srv = rospy.Service('set_cluster_iou_pct', SetInt, self.set_cluster_iou_pct)
        self.setGridcell_srv = rospy.Service('set_gridcell_size', SetFloat, self.set_gridcell_size_service)
        self.clear_srv = rospy.Service('clear_clouds', Trigger, self.clear_clouds_service)
        self.top1_cluster_srv = rospy.Service('get_top1_cluster', DynamicCluster, self.top1_cluster_service)
        self.marker_pub = rospy.Publisher('cluster_markers', MarkerArray, queue_size=5)
        self.draw_clusters_srv = rospy.Service('draw_pclouds', Trigger, self.draw_pclouds)
        self.evaluate_srv = rospy.Service('evaluate_detection', EvaluateDetection, self.evaluate_detection_service)
        
        # Evaluation results publisher
        self.eval_pub = rospy.Publisher('detection_evaluation', DetectionEvaluation, queue_size=5)
        
        # Dynamic Navigation Integration
        # Listen to navigation status to track if we're using dynamic navigation
        self.using_dynamic_nav = False
        self.nav_status_sub = rospy.Subscriber('/exploration_status', String, self.nav_status_callback)
        
        # Initialize CLIPSeg and OmDet models
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize CLIPSeg and OmDet models."""
        # Import models here to avoid loading them if not used
        try:
            from change_detection.clip_segmentation import clip_seg
            self.clipseg_model = clip_seg(self.query_list)
            rospy.loginfo("CLIPSeg model initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize CLIPSeg: {e}")
            self.clipseg_model = None
        
        try:
            # Use the actual OmDet implementation
            from change_detection.omdet_segmentation import omdet_segmentation
            self.omdet_model = omdet_segmentation(self.query_list)
            rospy.loginfo("OmDet model initialized")
        except Exception as e:
            rospy.logerr(f"Failed to initialize OmDet: {e}")
            self.omdet_model = None

    def pose_callback(self, msg):
        with self.pose_lock:
            self.pose_queue.append(msg)
            if len(self.pose_queue) > 10:
                self.pose_queue.pop(0)

    def cam_info_callback(self, msg):
        if self.pcloud_creator is None:
            self.pcloud_creator = pcloud_from_images(camera_params(msg.height, msg.width, msg.K[0], msg.K[4], msg.K[2], msg.K[5]))

    def rgbd_callback(self, rgb_msg, depth_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            if self.pcloud_creator:
                # Estimate motion if we have a previous frame
                self.estimate_motion(cv_image)
                
                for query in self.query_list:
                    # Process with both CLIPSeg and OmDet models
                    clipseg_mask, clipseg_probs = self.process_with_clipseg(cv_image, query)
                    omdet_mask, omdet_probs = self.process_with_omdet(cv_image, query)
                    
                    # Combine results from both models with temporal consistency
                    combined_mask, combined_probs = self.combine_results(clipseg_mask, clipseg_probs, omdet_mask, omdet_probs, query)
                    
                    # Update point cloud with combined results - for single query only
                    self.update_pcloud(query, combined_mask, combined_probs, clipseg_probs, omdet_probs, depth_image, cv_image, rgb_msg.header.stamp)
                    
                    # Store current results for next frame
                    self.prev_masks[query] = combined_mask.copy()
                    self.prev_probs[query] = combined_probs.copy()
                
                # Update last image and increment frame counter
                self.last_image = {"image": cv_image, "time": rgb_msg.header.stamp}
                self.frame_count += 1
                
        except CvBridgeError as e:
            print(e)

    def estimate_motion(self, current_frame):
        """Estimate motion between frames for temporal consistency."""
        if self.last_image is None:
            return
        
        try:
            # Convert to grayscale for optical flow
            prev_gray = cv2.cvtColor(self.last_image["image"], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow using Lucas-Kanade method
            # Parameters for lucas kanade optical flow
            lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            # Create grid of points to track
            height, width = prev_gray.shape
            step = 20  # Grid step size
            points = []
            for y in range(0, height, step):
                for x in range(0, width, step):
                    points.append([x, y])
            
            prev_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            
            # Calculate optical flow
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)
            
            # Select good points
            good_old = prev_points[status == 1]
            good_new = next_points[status == 1]
            
            if len(good_old) > 0 and len(good_new) > 0:
                # Calculate average motion vector
                motion = np.mean(good_new - good_old, axis=0)
                self.motion_vectors['x'] = motion[0]
                self.motion_vectors['y'] = motion[1]
                rospy.logdebug(f"Estimated motion: dx={motion[0]:.2f}, dy={motion[1]:.2f}")
            else:
                self.motion_vectors['x'] = 0
                self.motion_vectors['y'] = 0
                
        except Exception as e:
            rospy.logerr(f"Motion estimation error: {e}")
            self.motion_vectors['x'] = 0
            self.motion_vectors['y'] = 0

    def warp_previous_frame(self, prev_array, query):
        """Warp previous frame based on estimated motion."""
        if not self.motion_vectors:
            return prev_array
        
        try:
            # Create transform matrix for simple translation
            dx, dy = self.motion_vectors.get('x', 0), self.motion_vectors.get('y', 0)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            
            # Apply warping
            rows, cols = prev_array.shape
            warped = cv2.warpAffine(prev_array, M, (cols, rows))
            return warped
            
        except Exception as e:
            rospy.logerr(f"Warping error for {query}: {e}")
            return prev_array

    def process_with_clipseg(self, image, query):
        """Process image with CLIPSeg model for given query."""
        if self.clipseg_model is None:
            # Return empty results if model isn't initialized
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
        
        try:
            # Process image with CLIPSeg
            results = self.clipseg_model.process_query(image, query)
            
            # Get mask and probabilities
            # Adjust according to your actual CLIPSeg implementation
            probs = results['probs']  # Probability map
            mask = (probs > self.detection_threshold).astype(np.uint8) * 255
            
            return mask, probs
        except Exception as e:
            rospy.logerr(f"Error in CLIPSeg processing: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)

    def process_with_omdet(self, image, query):
        """Process image with OmDet model for given query."""
        if self.omdet_model is None:
            # Return empty results if model isn't initialized
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
        
        try:
            # Process image with actual OmDet model
            self.omdet_model.process_image_numpy(image)
            probs = self.omdet_model.get_prob_array(query)
            
            if probs is None:
                return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
                
            # Get mask from probability map
            mask = (probs > self.detection_threshold).astype(np.uint8) * 255
            
            return mask, probs
        except Exception as e:
            rospy.logerr(f"Error in OmDet processing: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)

    def combine_results(self, clipseg_mask, clipseg_probs, omdet_mask, omdet_probs, query=None):
        """
        Combine results from CLIPSeg and OmDet models using an advanced hybrid boosting strategy.
        Now includes temporal consistency for more stable detections across frames.
        
        This method implements:
        1. Adaptive weighting based on confidence
        2. Complementary strengths boosting (CLIPSeg for semantics, OmDet for boundaries)
        3. Cross-model confidence boosting
        4. Temporal consistency boosting
        """
        # Initialize combined arrays
        combined_probs = np.zeros_like(clipseg_probs, dtype=np.float32)
        combined_mask = np.zeros_like(clipseg_mask, dtype=np.uint8)
        
        # Calculate confidence for each model
        clipseg_conf = np.mean(clipseg_probs[clipseg_probs > self.detection_threshold]) if np.any(clipseg_probs > self.detection_threshold) else 0.5
        omdet_conf = np.mean(omdet_probs[omdet_probs > self.detection_threshold]) if np.any(omdet_probs > self.detection_threshold) else 0.5
        
        # Normalize confidences for adaptive weighting
        total_conf = clipseg_conf + omdet_conf
        if total_conf > 0:
            clipseg_weight = clipseg_conf / total_conf
            omdet_weight = omdet_conf / total_conf
        else:
            clipseg_weight = 0.6  # Default weights if no confident detections
            omdet_weight = 0.4
        
        # Get binary masks for high confidence regions in each model
        clipseg_high_conf = clipseg_probs > (self.detection_threshold + 0.1)
        omdet_high_conf = omdet_probs > (self.detection_threshold + 0.1)
        
        # Step 1: Initialize with adaptive weighted combination
        combined_probs = clipseg_weight * clipseg_probs + omdet_weight * omdet_probs
        
        # Step 2: Boundary refinement (if OmDet has high confidence near object edges)
        # Find edges in CLIPSeg mask (likely object boundaries)
        if np.any(clipseg_high_conf):
            # Create dilated and eroded versions to find boundary regions
            from scipy import ndimage
            clipseg_dilated = ndimage.binary_dilation(clipseg_high_conf, iterations=3)
            clipseg_eroded = ndimage.binary_erosion(clipseg_high_conf, iterations=3)
            boundary_region = clipseg_dilated & ~clipseg_eroded
            
            # In boundary regions, boost OmDet's influence if it has high confidence
            boundary_boost_mask = boundary_region & omdet_high_conf
            if np.any(boundary_boost_mask):
                # Give more weight to OmDet in boundary regions where it's confident
                boundary_boost = 0.8  # High weight to OmDet at boundaries
                combined_probs[boundary_boost_mask] = (1 - boundary_boost) * clipseg_probs[boundary_boost_mask] + boundary_boost * omdet_probs[boundary_boost_mask]
        
        # Step 3: Semantic region boosting (CLIPSeg often better at identifying the correct semantic regions)
        # In regions where CLIPSeg has very high confidence, boost its influence
        clipseg_very_high_conf = clipseg_probs > (self.detection_threshold + 0.2)
        if np.any(clipseg_very_high_conf):
            semantic_boost = 0.8  # High weight to CLIPSeg for semantic understanding
            combined_probs[clipseg_very_high_conf] = semantic_boost * clipseg_probs[clipseg_very_high_conf] + (1 - semantic_boost) * omdet_probs[clipseg_very_high_conf]
        
        # Step 4: Cross-model confidence boosting
        # If one model has high confidence where the other has medium confidence,
        # boost the medium confidence areas
        clipseg_med_conf = (clipseg_probs > (self.detection_threshold - 0.1)) & ~clipseg_high_conf
        omdet_med_conf = (omdet_probs > (self.detection_threshold - 0.1)) & ~omdet_high_conf
        
        # Where CLIPSeg is medium and OmDet is high, boost CLIPSeg
        boost_clipseg = clipseg_med_conf & omdet_high_conf
        if np.any(boost_clipseg):
            # Calculate boosted CLIPSeg probabilities
            boosted_clipseg = np.minimum(clipseg_probs[boost_clipseg] * 1.2, 1.0)
            combined_probs[boost_clipseg] = 0.5 * boosted_clipseg + 0.5 * omdet_probs[boost_clipseg]
        
        # Where OmDet is medium and CLIPSeg is high, boost OmDet
        boost_omdet = omdet_med_conf & clipseg_high_conf
        if np.any(boost_omdet):
            # Calculate boosted OmDet probabilities
            boosted_omdet = np.minimum(omdet_probs[boost_omdet] * 1.2, 1.0)
            combined_probs[boost_omdet] = 0.5 * clipseg_probs[boost_omdet] + 0.5 * boosted_omdet
        
        # Step 5: Temporal consistency boosting
        if self.consistency_enabled and query and query in self.prev_probs and self.frame_count > 0:
            # Warp previous frame to account for motion
            warped_prev_probs = self.warp_previous_frame(self.prev_probs[query], query)
            
            # Create a temporal consistency mask - areas where temporal info is reliable
            temporal_mask = warped_prev_probs > (self.detection_threshold - 0.05)
            
            # Gradually increase temporal weight over initial frames to avoid initial instability
            current_temporal_weight = min(self.temporal_weight, 0.1 * min(10, self.frame_count))
            
            # Only apply temporal consistency where we have reliable previous detection
            if np.any(temporal_mask):
                # Blend current and previous probabilities with temporal weight
                combined_probs[temporal_mask] = (1 - current_temporal_weight) * combined_probs[temporal_mask] + \
                                               current_temporal_weight * warped_prev_probs[temporal_mask]
                                               
                # Log stats about temporal consistency
                overlap_ratio = np.sum(temporal_mask & (combined_probs > self.detection_threshold)) / \
                               max(1, np.sum(combined_probs > self.detection_threshold))
                if overlap_ratio > 0:
                    rospy.logdebug(f"Temporal consistency overlap: {overlap_ratio:.2f} for {query}")
        
        # Apply threshold to get binary mask
        combined_mask = (combined_probs > self.detection_threshold).astype(np.uint8) * 255
        
        # Log the effectiveness of the boosting
        clipseg_pixels = np.sum(clipseg_mask > 0)
        omdet_pixels = np.sum(omdet_mask > 0)
        combined_pixels = np.sum(combined_mask > 0)
        
        if clipseg_pixels > 0 or omdet_pixels > 0:
            rospy.loginfo(f"Hybrid Boost - CLIPSeg: {clipseg_pixels} px, OmDet: {omdet_pixels} px, Combined: {combined_pixels} px")
            rospy.loginfo(f"Model weights - CLIPSeg: {clipseg_weight:.2f}, OmDet: {omdet_weight:.2f}")
        
        return combined_mask, combined_probs

    def update_pcloud(self, query, mask, probs, clipseg_probs, omdet_probs, depth_image, rgb_image, timestamp):
        """
        Update the point cloud for the given query with the combined results.
        Store both combined and individual model probabilities.
        """
        if self.pcloud_creator is None:
            return
        
        # Convert mask and depth to point cloud
        points, colors = self.pcloud_creator.rgbd_to_pcloud(depth_image, rgb_image)
        if points is None or len(points) == 0:
            return
        
        # Filter points based on mask
        valid_points = []
        valid_colors = []
        valid_probs = []
        valid_clipseg_probs = []
        valid_omdet_probs = []
        
        height, width = mask.shape
        for i in range(len(points)):
            x, y = int(points[i, 0]), int(points[i, 1])
            if 0 <= y < height and 0 <= x < width and mask[y, x] > 0:
                valid_points.append(points[i])
                valid_colors.append(colors[i])
                valid_probs.append(probs[y, x])
                valid_clipseg_probs.append(clipseg_probs[y, x])
                valid_omdet_probs.append(omdet_probs[y, x])
        
        if valid_points:
            # Convert to tensors
            new_points = torch.tensor(valid_points, dtype=torch.float, device=DEVICE)
            new_colors = torch.tensor(valid_colors, dtype=torch.float, device=DEVICE)
            new_probs = torch.tensor(valid_probs, dtype=torch.float, device=DEVICE)
            new_clipseg_probs = torch.tensor(valid_clipseg_probs, dtype=torch.float, device=DEVICE)
            new_omdet_probs = torch.tensor(valid_omdet_probs, dtype=torch.float, device=DEVICE)
            
            # Append to existing point cloud
            self.pcloud[query]['xyz'] = torch.cat((self.pcloud[query]['xyz'], new_points), dim=0)
            self.pcloud[query]['rgb'] = torch.cat((self.pcloud[query]['rgb'], new_colors), dim=0)
            self.pcloud[query]['probs'] = torch.cat((self.pcloud[query]['probs'], new_probs), dim=0)
            self.pcloud[query]['clipseg_probs'] = torch.cat((self.pcloud[query]['clipseg_probs'], new_clipseg_probs), dim=0)
            self.pcloud[query]['omdet_probs'] = torch.cat((self.pcloud[query]['omdet_probs'], new_omdet_probs), dim=0)

    def draw_pclouds(self, msg):
        resp = TriggerResponse()
        resp.success = True
        for query in self.query_list:
            # Get clusters for this query
            positive_clusters, pos_likelihoods = self.match_clusters(self.cluster_metric, query)
            
            if len(positive_clusters) < 1:
                rospy.loginfo(f"No clusters found for {query}")
                continue
            
            # Create a visualization message with detected object boxes
            viz_msg = MarkerArray()
            
            # Add boxes for each object
            for i, obj_ in enumerate(positive_clusters[:5]):  # Visualize top 5 clusters
                box_marker = self.create_box_marker(i, obj_, query)
                viz_msg.markers.append(box_marker)
            
            # Publish the visualization message for this query
            self.marker_pub.publish(viz_msg)
            
            rospy.loginfo(f"Published {len(viz_msg.markers)} markers for {query}")
        
        resp.message = f"Visualizing clusters for {len(self.query_list)} queries"
        return resp

    def match_clusters(self, metric, query):
        """
        Match clusters based on a specified metric.
        Now handles only one query at a time.
        """
        # Create clusters for the query
        if query not in self.pcloud:
            return ([], [])
        
        all_objects = create_object_clusters(self.pcloud[query]['xyz'].cpu().numpy(), 
                                            self.pcloud[query]['probs'].cpu().numpy(),
                                            self.pcloud[query]['clipseg_probs'].cpu().numpy(),
                                            self.pcloud[query]['omdet_probs'].cpu().numpy(),
                                            -1.0, self.detection_threshold)
        
        pos_likelihoods = []
        all_objects = sorted(all_objects, key=lambda x: -x.prob_stats[metric])
        
        for obj_ in all_objects:
            pos_likelihoods.append(obj_.prob_stats[metric])
        
        rospy.loginfo(f"Found {len(all_objects)} clusters for {query}")
        
        return (all_objects, pos_likelihoods)

    def create_box_marker(self, idx, obj_, label):
        """Create a marker for visualizing an object bounding box"""
        M = Marker()
        M.header.frame_id = "map"
        M.header.stamp = rospy.Time.now()
        M.ns = f"hybrid_objects_{label}"
        M.id = idx
        M.type = Marker.CUBE
        M.action = Marker.ADD
        M.pose.position.x = (obj_.box[0, 0] + obj_.box[1, 0]) / 2
        M.pose.position.y = (obj_.box[0, 1] + obj_.box[1, 1]) / 2
        M.pose.position.z = (obj_.box[0, 2] + obj_.box[1, 2]) / 2
        M.pose.orientation.w = 1.0
        M.scale.x = obj_.box[1, 0] - obj_.box[0, 0]
        M.scale.y = obj_.box[1, 1] - obj_.box[0, 1]
        M.scale.z = obj_.box[1, 2] - obj_.box[0, 2]
        
        # Set color based on object type
        if label == "chair":
            M.color.r = 0.0
            M.color.g = 1.0
            M.color.b = 0.0
        elif label == "table":
            M.color.r = 0.0
            M.color.g = 0.0
            M.color.b = 1.0
        elif label == "sofa":
            M.color.r = 1.0
            M.color.g = 0.0 
            M.color.b = 0.0
        else:
            M.color.r = 0.5
            M.color.g = 0.5
            M.color.b = 0.5
            
        M.color.a = 0.5
        return M

    def set_cluster_metric_service(self, req):
        """Set the clustering metric."""
        if req.value in ['hybrid-mean', 'clipseg-mean', 'omdet-mean', 'combo-mean']:
            print(f"Setting new cluster metric as {req.value}")
            self.cluster_metric = req.value
        else:
            print('Currently supported options include hybrid-mean, clipseg-mean, omdet-mean, combo-mean')
        return SetStringResponse()

    def set_cluster_size_service(self, req):
        """Set minimum cluster size."""
        self.cluster_min_points = req.value
        print(f"Changing minimum cluster size to {self.cluster_min_points} points")
        return SetIntResponse()

    def set_gridcell_size_service(self, req):
        """Set grid cell size for clustering."""
        self.gridcell_size = req.value
        print(f"Changing gridcell size to {self.gridcell_size} m")
        return SetFloatResponse()

    def set_cluster_iou_pct(self, req):
        """Set cluster IOU percentage threshold."""
        self.cluster_iou = req.value / 100.0
        print(f"Changing minimum cluster iou to {self.cluster_iou} pct")
        return SetIntResponse()

    def clear_clouds_service(self, msg):
        """Clear all point clouds."""
        resp = TriggerResponse()
        resp.success = True
        for query in self.query_list:
            self.pcloud[query] = {
                'xyz': torch.zeros((0,3), dtype=float, device=DEVICE),
                'probs': torch.zeros((0), dtype=float, device=DEVICE),
                'rgb': torch.zeros((0,3), dtype=float, device=DEVICE),
                'clipseg_probs': torch.zeros((0), dtype=float, device=DEVICE),
                'omdet_probs': torch.zeros((0), dtype=float, device=DEVICE)
            }
        self.last_image = None
        
        # Reset temporal consistency variables
        self.prev_masks = {}
        self.prev_probs = {}
        self.motion_vectors = {}
        self.frame_count = 0
        
        resp.message = "clouds cleared"
        return resp

    def top1_cluster_service(self, msg):
        target_type = msg.target_type
        rospy.loginfo("Retrieving top1 cluster for type: %s"%(target_type))
        
        if target_type not in self.query_list:
            rospy.logerr(f"Target type {target_type} not in query list {self.query_list}")
            resp = DynamicClusterResponse()
            resp.box_exists = False
            return resp

        query = target_type
        positive_clusters, pos_likelihoods = self.match_clusters(self.cluster_metric, query)
        
        if len(positive_clusters) < 1:
            rospy.loginfo(f"No clusters found for {query}")
            resp = DynamicClusterResponse()
            resp.box_exists = False
            return resp
        
        best_cluster = positive_clusters[0]
        
        resp = DynamicClusterResponse()
        resp.box_exists = True
        resp.box_min.x = best_cluster.box[0, 0]
        resp.box_min.y = best_cluster.box[0, 1]
        resp.box_min.z = best_cluster.box[0, 2]
        resp.box_max.x = best_cluster.box[1, 0]
        resp.box_max.y = best_cluster.box[1, 1]
        resp.box_max.z = best_cluster.box[1, 2]
        resp.confidence = pos_likelihoods[0]
        
        return resp

    def nav_status_callback(self, msg):
        """Callback for navigation status updates"""
        status = msg.data
        if "dynamic" in status.lower():
            self.using_dynamic_nav = True
            rospy.loginfo("Dynamic navigation detected and integrated with hybrid detection")

    def load_ground_truth(self, gt_file):
        """Load ground truth data from a pickle file for evaluation"""
        try:
            with open(gt_file, 'rb') as f:
                self.ground_truth = pickle.load(f)
            
            # Log what we loaded
            for query, boxes in self.ground_truth.items():
                if query in self.query_list:
                    rospy.loginfo(f"Loaded {len(boxes)} ground truth boxes for {query}")
                else:
                    rospy.logwarn(f"Ground truth for {query} loaded but not in query list")
            
            return True
        except Exception as e:
            rospy.logerr(f"Error loading ground truth: {e}")
            return False
            
    def evaluate_detection_service(self, req):
        """Service to evaluate current detections against ground truth"""
        resp = EvaluateDetectionResponse()
        
        # Check if we have ground truth data
        if not self.ground_truth:
            if req.ground_truth_file and os.path.exists(req.ground_truth_file):
                success = self.load_ground_truth(req.ground_truth_file)
                if not success:
                    resp.success = False
                    resp.message = "Failed to load ground truth data"
                    return resp
            else:
                resp.success = False
                resp.message = "No ground truth data available and no file provided"
                return resp
        
        # Perform evaluation
        evaluation = self.evaluate_against_groundtruth()
        
        # Create response
        resp.success = True
        resp.message = "Evaluation completed successfully"
        
        # Publish detailed results
        self.publish_evaluation_results(evaluation)
        
        return resp
    
    def publish_evaluation_results(self, evaluation):
        """Publish evaluation results as a ROS message"""
        msg = DetectionEvaluation()
        msg.header.stamp = rospy.Time.now()
        msg.using_dynamic_navigation = self.using_dynamic_nav
        msg.num_frames_processed = self.frame_count
        msg.detection_threshold = self.detection_threshold
        msg.temporal_weight = self.temporal_weight
        
        for query, metrics in evaluation.items():
            query_metrics = DetectionMetrics()
            query_metrics.query = query
            query_metrics.precision = metrics["precision"]
            query_metrics.recall = metrics["recall"]
            query_metrics.f1_score = metrics["f1_score"]
            query_metrics.true_positives = metrics["true_positives"]
            query_metrics.false_positives = metrics["false_positives"]
            query_metrics.false_negatives = metrics["false_negatives"]
            query_metrics.total_detections = metrics["total_detections"]
            query_metrics.total_points = metrics["total_points"]
            query_metrics.average_points_per_cluster = metrics["average_points_per_cluster"]
            query_metrics.point_cloud_density = metrics["point_cloud_density"]
            
            msg.metrics.append(query_metrics)
        
        self.eval_pub.publish(msg)
        
        # Log to console too
        rospy.loginfo("===== DETECTION EVALUATION RESULTS =====")
        rospy.loginfo(f"Using dynamic navigation: {self.using_dynamic_nav}")
        for query, metrics in evaluation.items():
            rospy.loginfo(f"Results for {query}:")
            rospy.loginfo(f"  Precision: {metrics['precision']:.4f}")
            rospy.loginfo(f"  Recall: {metrics['recall']:.4f}")
            rospy.loginfo(f"  F1-Score: {metrics['f1_score']:.4f}")
            rospy.loginfo(f"  Detections: {metrics['total_detections']} (TP: {metrics['true_positives']}, FP: {metrics['false_positives']})")
    
    def evaluate_against_groundtruth(self):
        """
        Evaluate detection results against ground truth.
        
        Returns:
            Dictionary of evaluation metrics
        """
        evaluation = {}
        
        for query in self.query_list:
            if query not in self.ground_truth:
                rospy.logwarn(f"No ground truth available for {query}")
                continue
            
            # Get point cloud for this query
            query_pcloud = self.pcloud[query]
            
            # Create clusters from the point cloud
            positive_clusters, _ = self.match_clusters(self.cluster_metric, query)
            
            # Extract detection bounding boxes
            detection_boxes = [obj_.box for obj_ in positive_clusters]
            
            # Get ground truth boxes for this query
            gt_boxes = self.ground_truth[query]
            
            # Calculate IoU between each detection and ground truth
            iou_threshold = 0.5  # Minimum IoU for a true positive
            
            tp = 0  # True positives
            fp = 0  # False positives
            
            matched_gt = set()
            
            for det_box in detection_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    # Calculate IoU - convert to the right format if needed
                    if len(gt_box) == 6:  # [min_x, min_y, min_z, max_x, max_y, max_z]
                        gt_min = np.array([gt_box[0], gt_box[1], gt_box[2]])
                        gt_max = np.array([gt_box[3], gt_box[4], gt_box[5]])
                    else:  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                        gt_min = np.array(gt_box[0])
                        gt_max = np.array(gt_box[1])
                    
                    # Intersection
                    intersection_min = np.maximum(det_box[0], gt_min)
                    intersection_max = np.minimum(det_box[1], gt_max)
                    
                    if np.any(intersection_min >= intersection_max):
                        iou = 0
                    else:
                        intersection_vol = np.prod(intersection_max - intersection_min)
                        det_vol = np.prod(det_box[1] - det_box[0])
                        gt_vol = np.prod(gt_max - gt_min)
                        union_vol = det_vol + gt_vol - intersection_vol
                        iou = intersection_vol / union_vol
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            # False negatives are ground truths without matches
            fn = len(gt_boxes) - len(matched_gt)
            
            # Calculate precision, recall, and F-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate point cloud density metrics
            total_points = query_pcloud['xyz'].shape[0]
            avg_points_per_cluster = sum(obj_.size() for obj_ in positive_clusters) / len(positive_clusters) if positive_clusters else 0
            
            # Store evaluation metrics
            evaluation[query] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'total_detections': len(detection_boxes),
                'total_ground_truth': len(gt_boxes),
                'total_points': total_points,
                'average_points_per_cluster': avg_points_per_cluster,
                'point_cloud_density': total_points / max(1, np.prod(query_pcloud['xyz'].max(dim=0)[0].cpu().numpy() - query_pcloud['xyz'].min(dim=0)[0].cpu().numpy())) if total_points > 0 else 0
            }
            
            # Update stored metrics
            self.eval_metrics[query] = evaluation[query]
            
        return evaluation

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('queries', type=str, nargs='*', default=None, help='Set of target classes to build point clouds for')
    parser.add_argument('--threshold', type=float, default=0.80, help='proposed detection threshold (default = 0.75)')
    parser.add_argument('--min_points', type=int, default=200, help='minimum number of points in a valid cluster')
    parser.add_argument('--save_dir', type=str, default='save_results', help='directory in which to store results')
    parser.add_argument('--travel_dist', type=float, default=0.1, help='minimum amount of travel distance to save new pclouds')
    parser.add_argument('--travel_angle', type=float, default=0.1, help='minimum angle of travel to save new pclouds')
    
    args = parser.parse_args()
    
    if args.queries is None or len(args.queries) == 0:
        print("Must specify at least one target object")
        exit(1)
    
    query_list = args.queries
    print(f"Building point clouds for {query_list}")
    
    travel_params = [args.travel_dist, args.travel_angle]
    hybrid = hybrid_query_localize(query_list, args.min_points, args.threshold, travel_params, args.save_dir)
    
    rospy.spin()