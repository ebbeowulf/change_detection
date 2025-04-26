import datetime
B1=datetime.datetime.now()
import torch
print(f"Library Load Time: {(datetime.datetime.now()-B1).total_seconds()}")
import pickle
import numpy as np
import cv2
import os
import pdb
import sys
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'change_detection', 'scripts'))
sys.path.append(scripts_path)
from change_detection.segmentation import image_segmentation
from rgbd_file_list import rgbd_file_list
from camera_params import camera_params
import copy
from sklearn.cluster import DBSCAN
from farthest_point_sampling.fps import farthest_point_sampling
import time

# Optimized DBSCAN parameters based on memory
DBSCAN_MIN_SAMPLES = 10
DBSCAN_GRIDCELL_SIZE = 0.005
DBSCAN_EPS = 0.025
CLUSTER_MIN_COUNT = 10
CLUSTER_PROXIMITY_THRESH = 0.3
CLUSTER_TOUCHING_THRESH = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pointcloud_open3d(xyz_points, rgb_points=None, max_num_points=2000000):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    if xyz_points.shape[0] < max_num_points:
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        if rgb_points is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb_points[:, [2, 1, 0]] / 255)
    else:
        rr = np.random.choice(np.arange(xyz_points.shape[0]), max_num_points)
        pcd.points = o3d.utility.Vector3dVector(xyz_points[rr, :])
        if rgb_points is not None:
            rgb2 = rgb_points[rr, :]
            pcd.colors = o3d.utility.Vector3dVector(rgb2[:, [2, 1, 0]] / 255)
    return pcd

TIME_STRUCT = {'count': 0, 'times': np.zeros((3,), dtype=float)}

def get_distinct_clusters(pcloud, gridcell_size=DBSCAN_GRIDCELL_SIZE, eps=DBSCAN_EPS, 
                          min_samples=DBSCAN_MIN_SAMPLES, cluster_min_count=CLUSTER_MIN_COUNT, 
                          floor_threshold=0.1):
    """Get distinct clusters from point cloud using DBSCAN clustering algorithm."""
    np_points = np.asarray(pcloud.points)
    
    # Remove points near floor level
    if floor_threshold is not None:
        not_ground = np_points[:, 2] > floor_threshold
        np_points = np_points[not_ground]
    
    if np_points.shape[0] < min_samples:
        return []
    
    # Scale points by gridcell size for better clustering
    scaled_points = np_points / gridcell_size
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_points)
    labels = clustering.labels_
    
    # Extract clusters
    unique_labels = set(labels)
    clusters = []
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        
        cluster_points = np_points[labels == label]
        if len(cluster_points) >= cluster_min_count:
            clusters.append(object_pcloud(cluster_points))
    
    return clusters

class object_pcloud:
    def __init__(self, pts, label:str=None, num_samples=1000, sample=True):
        self.box = np.vstack((pts.min(0), pts.max(0)))
        self.pts = pts
        self.pts_shape = self.pts.shape
        self.label = label
        self.farthestP = None
        if sample:
            self.sample_pcloud(num_samples)
        self.prob_stats = None
        self.centroid = self.pts.mean(0)
    
    def sample_pcloud(self, num_samples):
        """Sample point cloud using farthest point sampling."""
        if self.pts.shape[0] > num_samples:
            self.farthestP = farthest_point_sampling(torch.tensor(self.pts, dtype=torch.float32),
                                                    num_samples).cpu().numpy()
        else:
            self.farthestP = np.arange(self.pts.shape[0])
    
    def set_label(self, label):
        self.label = label
    
    def is_box_overlap(self, input_cloud, dimensions=[0, 1, 2], threshold=0.3):
        """Check if bounding boxes overlap."""
        minA = self.box[0, dimensions]
        maxA = self.box[1, dimensions]
        minB = input_cloud.box[0, dimensions]
        maxB = input_cloud.box[1, dimensions]
        
        # Calculate IOU
        deltaA = maxA - minA
        areaA = np.prod(deltaA)
        deltaB = maxB - minB
        areaB = np.prod(deltaB)
        
        isct_min = np.vstack((minA, minB)).max(0)
        isct_max = np.vstack((maxA, maxB)).min(0)
        isct_delta = isct_max - isct_min
        
        if (isct_delta <= 0).sum() > 0:
            return False
        
        areaI = np.prod(isct_delta)
        iou = areaI / (areaA + areaB - areaI)
        
        return iou > threshold
    
    def compute_cloud_distance(self, input_cloud):
        """Compute distance between two point clouds."""
        return np.linalg.norm(self.centroid - input_cloud.centroid)
    
    def is_above(self, input_cloud):
        """Check if this cloud is above the input cloud."""
        return self.box[0, 2] > input_cloud.box[1, 2]
    
    def estimate_probability(self, original_xyz, original_prob):
        """Estimate probability statistics for points in this cloud."""
        mean_prob = 0.0
        max_prob = 0.0
        min_prob = 1.0
        
        if len(original_xyz) == 0 or self.pts.shape[0] == 0:
            self.prob_stats = {'mean': mean_prob, 'max': max_prob, 'min': min_prob}
            return
        
        # Find the indices of points in the original cloud that are in this cluster
        indices = []
        for i in range(self.pts.shape[0]):
            pt = self.pts[i]
            dists = np.sum((original_xyz - pt) ** 2, axis=1)
            closest_idx = np.argmin(dists)
            if np.sqrt(dists[closest_idx]) < 1e-6:
                indices.append(closest_idx)
        
        if len(indices) > 0:
            probs = original_prob[indices]
            mean_prob = np.mean(probs)
            max_prob = np.max(probs)
            min_prob = np.min(probs)
        
        self.prob_stats = {'mean': mean_prob, 'max': max_prob, 'min': min_prob}
    
    def size(self):
        """Return the number of points in the cloud."""
        return self.pts.shape[0]
    
    def compress_object(self):
        """Compress the object by keeping only sampled points."""
        if self.farthestP is not None:
            self.pts = self.pts[self.farthestP]
            self.pts_shape = self.pts.shape
            self.farthestP = np.arange(self.pts.shape[0])

class pcloud_from_images:
    def __init__(self, params:camera_params):
        self.params = params
        self.rows = torch.tensor(np.tile(np.arange(params.height).reshape(params.height, 1), (1, params.width)) - params.cy, device=DEVICE)
        self.cols = torch.tensor(np.tile(np.arange(params.width), (params.height, 1)) - params.cx, device=DEVICE)
        self.rot_matrixT = torch.tensor(params.rot_matrix, device=DEVICE)
        self.loaded_image = None
        self.classifier_type = None
        
        # Initialize model classes if available
        try:
            from change_detection.clip_segmentation import clip_seg
            self.clipseg_model = None  # Will be initialized when needed
            self.has_clipseg = True
        except ImportError:
            self.has_clipseg = False
            
        try:
            from change_detection.omdet_segmentation import omdet_segmentation
            self.omdet_model = None  # Will be initialized when needed
            self.has_omdet = True
        except ImportError:
            self.has_omdet = False
    
    def load_image_from_file(self, fList:rgbd_file_list, image_key, max_distance=10.0):
        """Load image from file list."""
        colorI = cv2.imread(fList.get_color_fileName(image_key), -1)
        depthI = cv2.imread(fList.get_depth_fileName(image_key), -1)
        poseM = fList.get_pose(image_key)
        return self.load_image(colorI, depthI, poseM, str(image_key), max_distance)
    
    def load_image(self, colorI:np.ndarray, depthI:np.ndarray, poseM:np.ndarray, uid_key:str, max_distance=10.0):
        """Load RGB-D image and pose."""
        self.loaded_image = {
            'color': colorI,
            'depth': torch.tensor(depthI.astype('float') / 1000.0, device=DEVICE),
            'pose': torch.tensor(poseM, device=DEVICE),
            'uid': uid_key
        }
        
        # Generate 3D points
        self.loaded_image['x'] = self.cols * self.loaded_image['depth'] / self.params.fx
        self.loaded_image['y'] = self.rows * self.loaded_image['depth'] / self.params.fy
        
        # Create depth mask for valid points
        self.loaded_image['depth_mask'] = (self.loaded_image['depth'] > 1e-4) * (self.loaded_image['depth'] < max_distance)
        
        return True
    
    def rgbd_to_pcloud(self, depthI:np.ndarray, colorI:np.ndarray, max_distance=10.0):
        """Convert RGB-D image to point cloud."""
        depthT = torch.tensor(depthI.astype('float') / 1000.0, device=DEVICE)
        colorT = torch.tensor(colorI, device=DEVICE)
        
        # Generate X and Y coordinates
        x = self.cols * depthT / self.params.fx
        y = self.rows * depthT / self.params.fy
        
        # Create depth mask for valid points
        depth_mask = (depthT > 1e-4) * (depthT < max_distance)
        
        # Extract valid points
        pts = torch.stack([x[depth_mask], y[depth_mask], depthT[depth_mask], torch.ones_like(depthT[depth_mask])], dim=1)
        colors = colorT[depth_mask]
        
        return pts.cpu().numpy(), colors.cpu().numpy()
    
    def initialize_models(self, tgt_class_list):
        """Initialize models for the given target class list."""
        if self.has_clipseg:
            try:
                from change_detection.clip_segmentation import clip_seg
                self.clipseg_model = clip_seg(tgt_class_list)
                print("Initialized CLIPSeg model")
            except Exception as e:
                print(f"Error initializing CLIPSeg: {e}")
                self.clipseg_model = None
            
        if self.has_omdet:
            try:
                from change_detection.omdet_segmentation import omdet_segmentation
                self.omdet_model = omdet_segmentation(tgt_class_list)
                print("Initialized OmDet model")
            except Exception as e:
                print(f"Error initializing OmDet: {e}")
                self.omdet_model = None
    
    def process_with_clipseg(self, image, query):
        """Process image with CLIPSeg model for given query."""
        if not self.has_clipseg or self.clipseg_model is None:
            return np.zeros(image.shape[:2], dtype=np.float32)
        
        try:
            # Process image with CLIPSeg
            results = self.clipseg_model.process_query(image, query)
            return results['probs']  # Return probability map
        except Exception as e:
            print(f"Error in CLIPSeg processing: {e}")
            return np.zeros(image.shape[:2], dtype=np.float32)
    
    def process_with_omdet(self, image, query):
        """Process image with OmDet model for given query."""
        if not self.has_omdet or self.omdet_model is None:
            return np.zeros(image.shape[:2], dtype=np.float32)
        
        try:
            # Process image with actual OmDet model
            self.omdet_model.process_image_numpy(image)
            probs = self.omdet_model.get_prob_array(query)
            if probs is None:
                return np.zeros(image.shape[:2], dtype=np.float32)
            return probs
        except Exception as e:
            print(f"Error in OmDet processing: {e}")
            return np.zeros(image.shape[:2], dtype=np.float32)
    
    def hybrid_process(self, image, query, detection_threshold=0.5, model_weights=None):
        """
        Process image with hybrid approach using CLIPSeg and OmDet models.
        Args:
            image: RGB image to process
            query: Query text
            detection_threshold: Threshold for detection
            model_weights: Dictionary with weights for models, e.g. {'clipseg': 0.6, 'omdet': 0.4}
        Returns:
            Dictionary with mask, probabilities, and model-specific probabilities
        """
        if model_weights is None:
            model_weights = {'clipseg': 0.6, 'omdet': 0.4}
        
        # Process with CLIPSeg
        clipseg_probs = self.process_with_clipseg(image, query)
        
        # Process with OmDet
        omdet_probs = self.process_with_omdet(image, query)
        
        # Combine results using weighted average
        combined_probs = model_weights['clipseg'] * clipseg_probs + model_weights['omdet'] * omdet_probs
        
        # Create mask by thresholding
        mask = (combined_probs > detection_threshold).astype(np.uint8) * 255
        
        return {
            'mask': mask,
            'probs': combined_probs,
            'clipseg_probs': clipseg_probs,
            'omdet_probs': omdet_probs
        }
    
    def process_fList_hybrid(self, fList:rgbd_file_list, tgt_class_list:list, conf_threshold=0.5, model_weights=None):
        """
        Process file list with hybrid model approach.
        Args:
            fList: List of files to process
            tgt_class_list: List of target classes
            conf_threshold: Confidence threshold
            model_weights: Weights for different models
        Returns:
            Dictionary of point clouds for each target class
        """
        if not self.has_clipseg and not self.has_omdet:
            print("Error: No models available for hybrid processing")
            return {}
        
        # Initialize models if needed
        self.initialize_models(tgt_class_list)
        
        if model_weights is None:
            model_weights = {'clipseg': 0.6, 'omdet': 0.4}
        
        pcloud = {}
        for query in tgt_class_list:
            pcloud[query] = {
                'xyz': torch.zeros((0, 3), dtype=torch.float, device=DEVICE),
                'probs': torch.zeros((0), dtype=torch.float, device=DEVICE),
                'rgb': torch.zeros((0, 3), dtype=torch.float, device=DEVICE),
                'clipseg_probs': torch.zeros((0), dtype=torch.float, device=DEVICE),
                'omdet_probs': torch.zeros((0), dtype=torch.float, device=DEVICE)
            }
        
        # Process each image in the file list
        for key in fList.keys():
            self.load_image_from_file(fList, key)
            if self.loaded_image is None:
                continue
            
            # Process each target class
            for query in tgt_class_list:
                # Process with hybrid approach
                result = self.hybrid_process(self.loaded_image['color'], query, conf_threshold, model_weights)
                
                # Convert to point cloud
                if result['mask'].sum() > 0:
                    mask = result['mask'] > 0
                    valid_x = self.loaded_image['x'][mask]
                    valid_y = self.loaded_image['y'][mask]
                    valid_depth = self.loaded_image['depth'][mask]
                    
                    # Create 3D points
                    pts = torch.stack([valid_x, valid_y, valid_depth, torch.ones_like(valid_depth)], dim=1)
                    pts_rot = torch.matmul(self.loaded_image['pose'], pts.transpose(0, 1))
                    pts_rot = pts_rot[:3, :].transpose(0, 1)
                    
                    # Get RGB values
                    colors = torch.tensor(self.loaded_image['color'], device=DEVICE)[mask]
                    
                    # Get probabilities
                    combined_probs = torch.tensor(result['probs'][mask], device=DEVICE)
                    clipseg_probs = torch.tensor(result['clipseg_probs'][mask], device=DEVICE)
                    omdet_probs = torch.tensor(result['omdet_probs'][mask], device=DEVICE)
                    
                    # Append to point cloud
                    pcloud[query]['xyz'] = torch.cat((pcloud[query]['xyz'], pts_rot), dim=0)
                    pcloud[query]['probs'] = torch.cat((pcloud[query]['probs'], combined_probs), dim=0)
                    pcloud[query]['rgb'] = torch.cat((pcloud[query]['rgb'], colors), dim=0)
                    pcloud[query]['clipseg_probs'] = torch.cat((pcloud[query]['clipseg_probs'], clipseg_probs), dim=0)
                    pcloud[query]['omdet_probs'] = torch.cat((pcloud[query]['omdet_probs'], omdet_probs), dim=0)
        
        return pcloud