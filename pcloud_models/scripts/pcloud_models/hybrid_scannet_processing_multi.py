import numpy as np
import argparse
import glob
import pickle
import os
import datetime
import json
from rgbd_file_list import rgbd_file_list
from camera_params import camera_params
import map_utils_hybrid
import pdb
import torch
import cv2
from scipy import ndimage
from sklearn.metrics import precision_recall_fscore_support


def load_camera_info(info_file):
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            if line[-1]=='\n':
                line=line[:-1]
            (key, val) = line.split(" = ")
            if key=='sceneType':
                info_dict[key] = val
            elif key=='axisAlignment' or key=='colorToDepthExtrinsics':
                info_dict[key] = np.fromstring(val, sep=' ')
            elif key=='colorToDepthExtrinsics':
                info_dict[key] = np.fromstring(val, sep=' ')
            else:
                try:
                    info_dict[key] = float(val)
                except Exception as e:
                    info_dict[key] = val

    if 'axisAlignment' not in info_dict:
        rot_matrix = np.identity(4)
    else:
        rot_matrix = info_dict['axisAlignment'].reshape(4, 4)
    
    return camera_params(info_dict['colorHeight'], info_dict['colorWidth'],info_dict['fx_color'],info_dict['fy_color'],info_dict['mx_color'],info_dict['my_color'],rot_matrix)

def get_scene_type(info_file):
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            if line[-1]=='\n':
                line=line[:-1]
            (key, val) = line.split(" = ")
            if key=='sceneType':
                info_dict[key] = val
            elif key=='axisAlignment' or key=='colorToDepthExtrinsics':
                info_dict[key] = np.fromstring(val, sep=' ')
            elif key=='colorToDepthExtrinsics':
                info_dict[key] = np.fromstring(val, sep=' ')
            else:
                try:
                    info_dict[key] = float(val)
                except Exception as e:
                    info_dict[key] = val
    
    if 'sceneType' in info_dict:
        return info_dict['sceneType']
    return None

def read_scannet_pose(pose_fName):
    # Get the pose - 
    try:
        with open(pose_fName,'r') as fin:
            LNs=fin.readlines()
            pose=np.zeros((4,4),dtype=float)
            for r_idx,ln in enumerate(LNs):
                if ln[-1]=='\n':
                    ln=ln[:-1]
                p_split=ln.split(' ')
                for c_idx, val in enumerate(p_split):
                    pose[r_idx, c_idx]=float(val)
        return pose
    except Exception as e:
        return None
    
def is_new_image(last_poseM, new_poseM, travel_dist=0.05, travel_ang=0.05):
    if last_poseM is None:
        return True
    deltaP=last_poseM[:,3]-new_poseM[:,3]
    dist=np.sqrt((deltaP**2).sum())
    vec1=np.matmul(last_poseM[:3,:3],[1,0,0])
    vec2=np.matmul(new_poseM[:3,:3],[1,0,0])
    deltaAngle=np.arccos(np.dot(vec1,vec2))
    if dist>travel_dist or deltaAngle>travel_ang:
        print(f"Dist {dist}, angle {deltaAngle} - new")        
        return True
    return False

def build_file_structure(image_dir, save_dir, use_pose_filter=False):
    fList = rgbd_file_list(image_dir, image_dir, save_dir, use_pose_filter)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(image_dir+'/frame*.txt')
    # Sort by frame id - this is useful for filtering
    key_set={int(A.split('-')[1].split('.')[0]):A for A in txt_files}
    last_poseM=None
    for id in range(min(key_set.keys()),max(key_set.keys())+1):
        if id not in key_set:
            continue
        fName=key_set[id]
        try:
            ppts=fName.split('.')
            rootName=ppts[0].split('/')[-1]
            number=int(rootName.split('-')[-1])
            pose=read_scannet_pose(fName)
            if use_pose_filter:
                if is_new_image(last_poseM, pose):
                    fList.add_file(number,rootName+'.color.jpg',rootName+'.depth_reg.png')
                    fList.add_pose(number, pose)
                    last_poseM=pose
            else:
                fList.add_file(number,rootName+'.color.jpg',rootName+'.depth_reg.png')
                fList.add_pose(number, pose)
        except Exception as e:
            continue
    return fList

def create_object_clusters(xyz, probs, clipseg_probs, omdet_probs, floor_thresh=-1.0, 
                          detection_threshold=0.5, compress_clusters=False, 
                          gridcell_size=0.005, min_samples=10):
    """Create object clusters from point cloud."""
    # Create a mask for points above floor
    if floor_thresh > 0:
        above_floor = xyz[:, 2] > floor_thresh
        xyz = xyz[above_floor]
        probs = probs[above_floor]
        clipseg_probs = clipseg_probs[above_floor] if clipseg_probs is not None else None
        omdet_probs = omdet_probs[above_floor] if omdet_probs is not None else None
    
    # Get high probability points
    high_prob = probs >= detection_threshold
    high_prob_xyz = xyz[high_prob]
    
    if high_prob_xyz.shape[0] < min_samples:
        return []
    
    # Scale points for DBSCAN
    scaled_points = high_prob_xyz / gridcell_size
    
    # Apply DBSCAN clustering
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=5.0, min_samples=min_samples).fit(scaled_points)
    labels = clustering.labels_
    
    # Extract clusters
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Remove noise points
    
    objects = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_xyz = high_prob_xyz[cluster_mask]
        
        # Get original indices
        original_indices = np.where(high_prob)[0][cluster_mask]
        
        # Create object point cloud
        obj = {
            'xyz': cluster_xyz,
            'prob_stats': {
                'mean': np.mean(probs[original_indices]),
                'max': np.max(probs[original_indices]),
                'min': np.min(probs[original_indices])
            },
            'box': np.vstack([cluster_xyz.min(axis=0), cluster_xyz.max(axis=0)]),
            'centroid': cluster_xyz.mean(axis=0),
            'size': cluster_xyz.shape[0]
        }
        
        if clipseg_probs is not None:
            obj['clipseg_mean'] = np.mean(clipseg_probs[original_indices])
        
        if omdet_probs is not None:
            obj['omdet_mean'] = np.mean(omdet_probs[original_indices])
        
        objects.append(obj)
    
    return objects

class HybridScannetProcessor:
    """Processor for ScanNet data using hybrid CLIPSeg and OmDet with temporal consistency"""
    
    def __init__(self, target_queries, threshold=0.8, temporal_weight=0.3):
        self.target_queries = target_queries
        self.detection_threshold = threshold
        self.temporal_weight = temporal_weight
        
        # Temporal consistency variables
        self.prev_masks = {}
        self.prev_probs = {}
        self.motion_vectors = {}
        self.last_image = None
        self.frame_count = 0
        self.consistency_enabled = True
        
        # Evaluation metrics
        self.eval_metrics = {}
        for query in target_queries:
            self.eval_metrics[query] = {
                'tp': 0,  # True positives
                'fp': 0,  # False positives
                'fn': 0,  # False negatives
                'total_points': 0,  # Total points in point cloud
                'cluster_counts': [],  # Number of points in each cluster
                'confidences': []  # Confidence scores for detections
            }
        
        # Initialize models
        try:
            from change_detection.clip_segmentation import clip_seg
            self.clipseg_model = clip_seg()
            print("CLIPSeg model initialized")
        except Exception as e:
            print(f"Failed to initialize CLIPSeg: {e}")
            self.clipseg_model = None
            
        try:
            from change_detection.omdet_segmentation import omdet_segmentation
            self.omdet_model = omdet_segmentation(self.target_queries)
            print("OmDet model initialized")
        except Exception as e:
            print(f"Failed to initialize OmDet: {e}")
            self.omdet_model = None
    
    def process_with_clipseg(self, image, query):
        """Process image with CLIPSeg model for given query."""
        if self.clipseg_model is None:
            # Return empty results if model isn't initialized
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
        
        try:
            # Process with CLIPSeg
            preds = self.clipseg_model.process_image_numpy(image)
            probs = self.clipseg_model.get_prob_array(query, preds)
            mask = (probs > self.detection_threshold).astype(np.uint8) * 255
            return mask, probs
        except Exception as e:
            print(f"Error in CLIPSeg processing: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)

    def process_with_omdet(self, image, query):
        """Process image with OmDet model for given query."""
        if self.omdet_model is None:
            # Return empty results if model isn't initialized
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
        
        try:
            # Process with OmDet
            preds = self.omdet_model.process_image_numpy(image)
            probs = self.omdet_model.get_prob_array(query, preds)
            mask = (probs > self.detection_threshold).astype(np.uint8) * 255
            return mask, probs
        except Exception as e:
            print(f"Error in OmDet processing: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8), np.zeros(image.shape[:2], dtype=np.float32)
    
    def estimate_motion(self, current_frame):
        """Estimate motion between frames for temporal consistency."""
        if self.last_image is None:
            return
        
        try:
            # Convert to grayscale for optical flow
            prev_gray = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
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
                print(f"Estimated motion: dx={motion[0]:.2f}, dy={motion[1]:.2f}")
            else:
                self.motion_vectors['x'] = 0
                self.motion_vectors['y'] = 0
                
        except Exception as e:
            print(f"Motion estimation error: {e}")
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
            print(f"Warping error for {query}: {e}")
            return prev_array
    
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
                    print(f"Temporal consistency overlap: {overlap_ratio:.2f} for {query}")
        
        # Apply threshold to get binary mask
        combined_mask = (combined_probs > self.detection_threshold).astype(np.uint8) * 255
        
        # Log the effectiveness of the boosting
        clipseg_pixels = np.sum(clipseg_mask > 0)
        omdet_pixels = np.sum(omdet_mask > 0)
        combined_pixels = np.sum(combined_mask > 0)
        
        if clipseg_pixels > 0 or omdet_pixels > 0:
            print(f"Hybrid Boost - CLIPSeg: {clipseg_pixels} px, OmDet: {omdet_pixels} px, Combined: {combined_pixels} px")
            print(f"Model weights - CLIPSeg: {clipseg_weight:.2f}, OmDet: {omdet_weight:.2f}")
        
        return combined_mask, combined_probs
    
    def evaluate_against_groundtruth(self, pcloud, ground_truth):
        """
        Evaluate detection results against ground truth.
        
        Args:
            pcloud: Dictionary of point clouds for each target
            ground_truth: Dictionary of ground truth boxes {query: [box1, box2, ...]}
                          where each box is [min_x, min_y, min_z, max_x, max_y, max_z]
        
        Returns:
            Dictionary of evaluation metrics
        """
        evaluation = {}
        
        for query in pcloud:
            if query not in ground_truth:
                print(f"No ground truth available for {query}")
                continue
            
            # Get point cloud for this query
            query_pcloud = pcloud[query]
            
            # Create clusters from the point cloud
            clusters = create_object_clusters(
                query_pcloud['xyz'].cpu().numpy(),
                query_pcloud['probs'].cpu().numpy(),
                query_pcloud['clipseg_probs'].cpu().numpy(),
                query_pcloud['omdet_probs'].cpu().numpy(),
                floor_thresh=-1.0,
                detection_threshold=self.detection_threshold
            )
            
            # Extract detection bounding boxes
            detection_boxes = [cluster['box'] for cluster in clusters]
            
            # Get ground truth boxes for this query
            gt_boxes = ground_truth[query]
            
            # Calculate IoU between each detection and ground truth
            iou_threshold = 0.5  # Minimum IoU for a true positive
            
            tp = 0  # True positives
            fp = 0  # False positives
            fn = 0  # False negatives
            
            matched_gt = set()
            
            for det_box in detection_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    # Calculate IoU
                    gt_min = np.array([gt_box[0], gt_box[1], gt_box[2]])
                    gt_max = np.array([gt_box[3], gt_box[4], gt_box[5]])
                    
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
            total_points = len(query_pcloud['xyz'])
            avg_points_per_cluster = np.mean([cluster['size'] for cluster in clusters]) if clusters else 0
            
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
                'point_cloud_density': total_points / (np.prod(query_pcloud['xyz'].max(dim=0)[0].cpu().numpy() - query_pcloud['xyz'].min(dim=0)[0].cpu().numpy())) if total_points > 0 else 0
            }
            
            print(f"Evaluation for {query}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1_score:.4f}")
            print(f"  Detections: {len(detection_boxes)} (TP: {tp}, FP: {fp})")
            print(f"  Ground Truth: {len(gt_boxes)} (FN: {fn})")
            print(f"  Average points per cluster: {avg_points_per_cluster:.1f}")
            
        return evaluation
    
    def process_dataset(self, fList, pcloud_init, ground_truth=None):
        """Process the entire dataset with hybrid models and temporal consistency"""
        # Initialize empty point cloud for each query
        pcloud = {}
        for query in self.target_queries:
            pcloud[query] = {
                'xyz': torch.zeros((0,3), dtype=float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                'probs': torch.zeros((0), dtype=float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                'rgb': torch.zeros((0,3), dtype=float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                'clipseg_probs': torch.zeros((0), dtype=float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                'omdet_probs': torch.zeros((0), dtype=float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            }
        
        # Process images in order
        image_keys = sorted(fList.rgb_list.keys())
        for key in image_keys:
            try:
                # Load image and depth
                rgb_image = cv2.imread(fList.rgb_list[key])
                depth_image = cv2.imread(fList.depth_list[key], cv2.IMREAD_ANYDEPTH)
                
                if rgb_image is None or depth_image is None:
                    print(f"Error loading images for key {key}")
                    continue
                
                # Estimate motion from previous frame
                self.estimate_motion(rgb_image)
                
                # Process each query
                for query in self.target_queries:
                    # Process with both models
                    clipseg_mask, clipseg_probs = self.process_with_clipseg(rgb_image, query)
                    omdet_mask, omdet_probs = self.process_with_omdet(rgb_image, query)
                    
                    # Combine with temporal consistency
                    combined_mask, combined_probs = self.combine_results(
                        clipseg_mask, clipseg_probs, 
                        omdet_mask, omdet_probs, 
                        query
                    )
                    
                    # Update point cloud
                    new_pcloud = pcloud_init.process_image(
                        key, 
                        combined_mask, 
                        combined_probs, 
                        clipseg_probs, 
                        omdet_probs,
                        fList
                    )
                    
                    if new_pcloud is not None:
                        # Combine with existing point cloud
                        for k in new_pcloud.keys():
                            if k in pcloud[query]:
                                pcloud[query][k] = torch.cat((pcloud[query][k], new_pcloud[k]), dim=0)
                    
                    # Store current results for next frame
                    self.prev_masks[query] = combined_mask.copy()
                    self.prev_probs[query] = combined_probs.copy()
                
                # Update last image and frame counter
                self.last_image = rgb_image.copy()
                self.frame_count += 1
                
                print(f"Processed frame {key} ({self.frame_count}/{len(image_keys)})")
                
            except Exception as e:
                print(f"Error processing frame {key}: {e}")
        
        # Calculate evaluation metrics if ground truth is provided
        evaluation = None
        if ground_truth is not None:
            evaluation = self.evaluate_against_groundtruth(pcloud, ground_truth)
        
        return pcloud, evaluation

def load_scannet_ground_truth(scene_id, root_dir, target_classes):
    """
    Load ScanNet ground truth annotations directly.
    
    Args:
        scene_id: Scene ID (e.g., 'scene0000_00')
        root_dir: Root directory of ScanNet data
        target_classes: List of target classes to extract
        
    Returns:
        Dictionary of ground truth boxes in our format
    """
    # Define key paths
    annotation_path = os.path.join(root_dir, scene_id, f"{scene_id}_vh_clean_2.0.aggregation.json")
    segments_path = os.path.join(root_dir, scene_id, f"{scene_id}_vh_clean_2.0.segs.json")
    mesh_path = os.path.join(root_dir, scene_id, f"{scene_id}_vh_clean_2.ply")
    
    print(f"Loading ScanNet ground truth from {annotation_path}")
    
    # Check if files exist
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file not found at {annotation_path}")
        return None
    
    # Try to find the labels mapping file
    labels_path = None
    possible_paths = [
        os.path.join(root_dir, "scannetv2-labels.combined.tsv"),
        os.path.join(root_dir, "..", "scannetv2-labels.combined.tsv"),
        os.path.join(root_dir, "..", "..", "scannetv2-labels.combined.tsv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            labels_path = path
            break
    
    if labels_path is None:
        print("Warning: ScanNet labels file not found. Using NYU40 class IDs directly.")
    
    # Try to load the label mapping
    label_map = {}
    nyu40_to_label = {}
    if labels_path:
        try:
            with open(labels_path, 'r') as f:
                for line in f.readlines()[1:]:  # Skip header
                    parts = line.strip().split('\t')
                    if len(parts) >= 8:
                        raw_label = parts[1].lower()  # Raw category text
                        nyu40_id = int(parts[4])      # NYU40 ID
                        nyu40_name = parts[7].lower() # NYU40 category name
                        
                        # Map NYU40 IDs to readable class names
                        nyu40_to_label[nyu40_id] = nyu40_name
            print(f"Loaded {len(nyu40_to_label)} class mappings from {labels_path}")
        except Exception as e:
            print(f"Error loading label mappings: {e}")
    
    # Load aggregation file (instance segmentation)
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return None
    
    # Create the ground truth dictionary
    gt_boxes = {}
    for target_class in target_classes:
        gt_boxes[target_class] = []
    
    # Process each object instance
    for segment_group in annotations['segGroups']:
        # Get the object label and ID
        label_id = segment_group['label']
        instance_id = segment_group['objectId']
        
        # Convert label to our format
        object_class = None
        
        # Try to get the class name from the label mapping
        if label_id.isdigit():
            nyu40_id = int(label_id)
            if nyu40_id in nyu40_to_label:
                object_class = nyu40_to_label[nyu40_id]
        else:
            # The label might already be a text class
            object_class = label_id.lower()
        
        # Skip if not in our target classes
        if object_class not in target_classes:
            continue
        
        # Get the object's 3D bounding box
        # ScanNet provides [minx, miny, minz, maxx, maxy, maxz]
        bbox = segment_group.get('bbox')
        if bbox:
            # Add the bounding box to our ground truth
            gt_boxes[object_class].append(bbox)
    
    # Count total objects found
    total_objects = sum(len(boxes) for boxes in gt_boxes.values())
    if total_objects == 0:
        print("Warning: No objects found in the ground truth for the specified classes.")
        print(f"Available classes in the scene: {[seg['label'] for seg in annotations['segGroups']]}")
    else:
        for cls, boxes in gt_boxes.items():
            print(f"Found {len(boxes)} instances of {cls}")
    
    return gt_boxes


def get_scannet_scene_id(root_dir):
    """
    Extract the ScanNet scene ID from the directory path.
    
    Args:
        root_dir: Directory path for a ScanNet scene
        
    Returns:
        Scene ID string or None if not found
    """
    # Try to extract scene ID from the path
    path_parts = root_dir.split(os.sep)
    
    # Look for a directory name that matches the pattern 'scene\d+_\d+'
    for part in path_parts:
        if part.startswith('scene') and '_' in part:
            parts = part.split('_')
            if len(parts) >= 2 and parts[0].startswith('scene') and parts[0][5:].isdigit() and parts[1].isdigit():
                return part
    
    # If no match is found in the path, try to find scene info file
    scene_info_file = os.path.join(root_dir, "scene_info.txt")
    if os.path.exists(scene_info_file):
        with open(scene_info_file, 'r') as f:
            for line in f:
                if line.startswith("scene"):
                    return line.strip()
    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    parser.add_argument('--param_file',type=str,default=None,help='camera parameter file for this scene - default is of form <raw_dir>/scene????_??.txt')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--threshold',type=float,default=0.8, help='proposed detection threshold (default = 0.8)')
    parser.add_argument('--temporal_weight',type=float,default=0.3, help='weight for temporal consistency (default = 0.3)')
    parser.add_argument('--pose_filter', dest='pose_filter', action='store_true')
    parser.add_argument('--gt_file', type=str, default=None, help='ground truth file for evaluation (in pickle format)')
    parser.add_argument('--use_scannet_gt', action='store_true', help='directly use ScanNet ground truth instead of loading from a pickle file')
    parser.set_defaults(pose_filter=False, use_scannet_gt=False)
    
    args = parser.parse_args()
    
    # Get the target objects from command-line arguments
    if not args.targets:
        print("Must specify at least one target with --targets")
        exit(1)
        
    target_queries = args.targets
    print(f"Target objects to detect: {target_queries}")

    save_dir = args.root_dir + "/" + args.save_dir
    fList = build_file_structure(args.root_dir + "/" + args.raw_dir, save_dir, args.pose_filter)
    if args.param_file is not None:
        par_file = args.param_file
    else:
        s_root = args.root_dir.split('/')
        if s_root[-1] == '':
            par_file = args.root_dir + "%s.txt" % (s_root[-2])
        else:
            par_file = args.root_dir + "/%s.txt" % (s_root[-1])
    params = load_camera_info(par_file)
    
    # Initialize the point cloud processor
    pcloud_init = map_utils_hybrid.pcloud_from_images(params)
    
    # Initialize hybrid processor with temporal consistency
    processor = HybridScannetProcessor(
        target_queries=target_queries,
        threshold=args.threshold,
        temporal_weight=args.temporal_weight
    )
    
    # Load ground truth
    ground_truth = None
    
    # Option 1: Load directly from ScanNet annotations
    if args.use_scannet_gt:
        scene_id = get_scannet_scene_id(args.root_dir)
        if scene_id:
            # The root_dir for ground truth is typically the scans directory
            # We'll try to guess it by going up from the raw_dir
            gt_root_dir = os.path.abspath(os.path.join(args.root_dir, ".."))
            if not os.path.exists(os.path.join(gt_root_dir, scene_id)):
                # Try another common path structure
                gt_root_dir = os.path.abspath(os.path.join(args.root_dir, "..", ".."))
            
            ground_truth = load_scannet_ground_truth(scene_id, gt_root_dir, target_queries)
            
            if ground_truth:
                print(f"Successfully loaded ScanNet ground truth for {scene_id}")
                
                # Save the ground truth for future use
                gt_save_path = os.path.join(save_dir, f"{scene_id}_gt.pickle")
                with open(gt_save_path, 'wb') as f:
                    pickle.dump(ground_truth, f)
                print(f"Saved ground truth to {gt_save_path} for future use")
            else:
                print(f"Could not load ScanNet ground truth for {scene_id}")
    
    # Option 2: Load from provided pickle file (falls back to this if ScanNet GT failed)
    if ground_truth is None and args.gt_file and os.path.exists(args.gt_file):
        print(f"Loading ground truth from {args.gt_file}")
        try:
            with open(args.gt_file, 'rb') as f:
                ground_truth = pickle.load(f)
        except Exception as e:
            print(f"Error loading ground truth: {e}")
    
    # Process the dataset
    pcloud, evaluation = processor.process_dataset(fList, pcloud_init, ground_truth)
    
    # Move tensors to CPU before saving
    for query in pcloud:
        for key in pcloud[query]:
            if isinstance(pcloud[query][key], torch.Tensor):
                pcloud[query][key] = pcloud[query][key].cpu()
    
    # Add metadata to the saved file
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'detection_threshold': args.threshold,
        'temporal_weight': args.temporal_weight,
        'query_list': target_queries,
        'num_frames_processed': processor.frame_count,
        'dataset_path': args.root_dir,
        'ground_truth_file': args.gt_file
    }
    
    # Save both point cloud, evaluation metrics, and metadata
    save_data = {
        'pcloud': pcloud,
        'metadata': metadata,
    }
    
    if evaluation:
        save_data['evaluation'] = evaluation
    
    print(f"Saving results to {save_dir}/hybrid_pclouds.pickle")
    with open(f"{save_dir}/hybrid_pclouds.pickle", "wb") as f:
        pickle.dump(save_data, f)
    
    # If we have evaluation results, also save them separately as CSV for easy analysis
    if evaluation:
        print(f"Saving evaluation metrics to {save_dir}/evaluation_metrics.csv")
        import csv
        with open(f"{save_dir}/evaluation_metrics.csv", 'w', newline='') as csvfile:
            fieldnames = ['query', 'precision', 'recall', 'f1_score', 'true_positives', 
                          'false_positives', 'false_negatives', 'total_detections', 
                          'total_ground_truth', 'total_points', 'average_points_per_cluster',
                          'point_cloud_density']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for query, metrics in evaluation.items():
                row = {'query': query}
                row.update(metrics)
                writer.writerow(row)
                
        print("Evaluation metrics saved. Use these for comparison with other models/approaches.")
