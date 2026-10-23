"""
Point cloud and clustering utilities for the change pipeline.
  get_rotated_points            depth pixels -> 3D points in the map frame
  voxel_downsample_sum          merge points into voxels, summing their weights
  get_weighted_clusters         weighted DBSCAN on voxels -> object_pcloud clusters
  object_pcloud                 one cluster: points, 3D box, centroid, score stats, projection into images
  identify_related_images_*     which images view a 3D location
  pointcloud_open3d, visualize_combined_xyzrgb   visualization helpers (used by visualize_colmap.py)
"""
import torch
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from change_pcloud_utils.rgbd_file_list import rgbd_file_list
from change_pcloud_utils.camera_params import camera_params
from change_pcloud_utils.farthest_point_sampling.fps import farthest_point_sampling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an open3d pointcloud object - will randomly sample the cloud
#   to reduce the number of points as necessary
def pointcloud_open3d(xyz_points,
                      rgb_points=None,
                      max_num_points=2000000
                      ):
    import open3d as o3d
    pcd=o3d.geometry.PointCloud()
    if xyz_points.shape[0]<max_num_points:
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        if rgb_points is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb_points[:,[2,1,0]]/255) 
    else:
        rr=np.random.choice(np.arange(xyz_points.shape[0]),max_num_points)
        pcd.points = o3d.utility.Vector3dVector(xyz_points[rr,:])
        if rgb_points is not None:
            rgb2=rgb_points[rr,:]
            pcd.colors = o3d.utility.Vector3dVector(rgb2[:,[2,1,0]]/255) 

    return pcd

# Combine together multiple point clouds into a single
#   cloud and display the result using open3d
def visualize_combined_xyzrgb(fList:rgbd_file_list, params:camera_params, howmany_files=100, skip=0, max_num_points=2000000, max_depth=10.0):
    rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
    cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)

    combined_xyz=np.zeros((0,3),dtype=float)
    combined_rgb=np.zeros((0,3),dtype=np.uint8)

    rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)

    howmany=0
    count=0
    for key in range(max(fList.keys())):
        if not fList.is_key(key):
            continue
        count+=1
        if count<=skip:
            continue
        count=0
        # Create the generic depth data
        print(fList.get_color_fileName(key))
        try:
            colorI=cv2.imread(fList.get_color_fileName(key), -1)
            depthI=cv2.imread(fList.get_depth_fileName(key), -1)
            depthT=torch.tensor(depthI.astype('float')/1000.0,device=DEVICE)
            colorT=torch.tensor(colorI,device=DEVICE)
        except Exception as e:
            print("File not found: " +str(e))
            continue
        x = cols*depthT/params.fx
        y = rows*depthT/params.fy
        depth_mask=(depthT>1e-4)*(depthT<max_depth)

        # Rotate the points into the right space
        # M=torch.matmul(rot_matrixT,torch.tensor(fList.get_pose(key),device=DEVICE))
        M=torch.tensor(fList.get_pose(key),device=DEVICE)
        pts=torch.stack([x[depth_mask],y[depth_mask],depthT[depth_mask],torch.ones(((depth_mask>0).sum()),device=DEVICE)],dim=1)
        pts_rot=torch.matmul(M,pts.transpose(0,1))
        pts_rot=pts_rot[:3,:].transpose(0,1)

        if pts_rot.shape[0]>100:
            combined_xyz=np.vstack((combined_xyz,pts_rot.cpu().numpy()))
            combined_rgb=np.vstack((combined_rgb,colorT[depth_mask].cpu().numpy()))
        howmany+=1
        if howmany==howmany_files:
            break
    return pointcloud_open3d(combined_xyz,rgb_points=combined_rgb, max_num_points=max_num_points)


def get_rotated_points(x, y, depth, filtered_maskT, rot_matrix):
    filtered_maskT = filtered_maskT.bool()
    pts=torch.stack([x[filtered_maskT],
                        y[filtered_maskT],
                        depth[filtered_maskT],
                        torch.ones(((filtered_maskT>0).sum()),
                        device=DEVICE)],dim=1)
    pts_rot=torch.matmul(rot_matrix,pts.transpose(0,1))
    return pts_rot[:3,:].transpose(0,1)


def voxel_downsample_sum(points: torch.Tensor, weights: torch.Tensor, voxel_size: float):
    """Merge points into voxels of size voxel_size. Returns one point per occupied voxel (the mean position
    of its points) and that voxel's weight (the SUM of its points' weights)."""
    weights = weights.to(dtype=points.dtype, device=points.device)
    idx = torch.floor(points / voxel_size).to(torch.int64)
    idx = idx - idx.min(dim=0).values            # non-negative indices, so the key below is unique per voxel
    span = idx.max(dim=0).values + 1
    keys = idx[:, 0] * span[1] * span[2] + idx[:, 1] * span[2] + idx[:, 2]
    unique_keys, inverse = torch.unique(keys, return_inverse=True)

    n = len(unique_keys)
    voxel_pts = torch.zeros((n, 3), dtype=points.dtype, device=points.device)
    voxel_weights = torch.zeros(n, dtype=points.dtype, device=points.device)
    voxel_pts.index_add_(0, inverse, points)
    voxel_weights.index_add_(0, inverse, weights)
    counts = torch.bincount(inverse, minlength=n).to(points.dtype)
    voxel_pts /= counts[:, None]
    return voxel_pts, voxel_weights


def get_weighted_clusters(points: torch.Tensor, weights: torch.Tensor,
                          voxel_size: float, eps: float, min_samples: float, min_cluster_voxels: int):
    """Cluster candidate points into objects.
      1. merge points into voxels (voxel weight = sum of point weights)
      2. weighted DBSCAN on the voxels: a voxel is a core voxel if the summed weight of the voxels
         within eps meters (itself included) is at least min_samples
      3. keep clusters with at least min_cluster_voxels voxels, largest first
    Each cluster's score statistics are computed from the original (not voxelized) points in its 3D box."""
    if points.shape[0] == 0:
        return []
    voxel_pts, voxel_weights = voxel_downsample_sum(points, weights, voxel_size)
    pts = voxel_pts.cpu().numpy()
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=10).fit(pts, sample_weight=voxel_weights.cpu().numpy()).labels_

    sizes = np.bincount(labels[labels >= 0]) if (labels >= 0).any() else np.zeros(0, dtype=int)
    clusters = []
    for label in np.argsort(-sizes):
        if sizes[label] < min_cluster_voxels:
            break
        cluster = object_pcloud(pts[labels == label], sample=False)
        cluster.estimate_probability(points, weights)
        clusters.append(cluster)
    return clusters


class object_pcloud():
    def __init__(self, pts, label:str=None, num_samples=1000, sample=True):
        self.box=np.vstack((pts.min(0),pts.max(0)))
        self.pts=pts
        self.pts_shape=self.pts.shape
        self.label=label
        self.farthestP=None
        if sample:
            self.sample_pcloud(num_samples)
        self.prob_stats=None
        self.centroid=self.pts.mean(0)

    def sample_pcloud(self, num_samples):
            self.farthestP=farthest_point_sampling(self.pts, num_samples)

    def set_label(self, label):
        self.label=label
    
    def is_box_overlap(self, input_cloud, dimensions=[0,1,2], threshold=0.3):
        for dim in dimensions:
            if self.box[1,dim]<(input_cloud.box[0,dim]-threshold) or self.box[0,dim]>=(input_cloud.box[1,dim]+threshold):
                return False
        return True

    def compute_cloud_distance(self, input_cloud):
        input_pt_matrix=input_cloud.pts[input_cloud.farthestP]
        min_sq_dist=1e10
        for pid in self.farthestP:
            min_sq_dist=min(min_sq_dist, ((input_pt_matrix-self.pts[pid])**2).sum(1).min())
        return np.sqrt(min_sq_dist)
    
    def is_above(self, input_cloud):
        # Should be overlapped in x + y directions
        if self.centroid[0]>input_cloud.box[0,0] and self.centroid[0]<input_cloud.box[1,0] and self.centroid[1]>input_cloud.box[0,1] and self.centroid[1]<input_cloud.box[1,1]:
            # Should also be "above" the other centroid
            return self.centroid[2]>input_cloud.centroid[2]
        return False
    
    def filter_points_in_box(self, xyz):
        filt=(xyz[:,0]>=self.box[0][0])*(xyz[:,0]<=self.box[1][0])*(xyz[:,1]>=self.box[0][1])*(xyz[:,1]<=self.box[1][1])*(xyz[:,2]>=self.box[0][2])*(xyz[:,2]<=self.box[1][2])
        return filt

    def estimate_probability(self, original_xyz, original_prob):
        filt=self.filter_points_in_box(original_xyz)
        self.prob_stats=dict()
        self.prob_stats['max']=original_prob[filt].max().to('cpu').item()
        self.prob_stats['mean']=original_prob[filt].mean().to('cpu').item()
        self.prob_stats['prob_sum']=original_prob[filt].sum().to('cpu').item()
        self.prob_stats['pcount']=filt.sum()
        self.prob_stats['stdev']=original_prob[filt].std().to('cpu').item()

        # Inverse stats
        # self.prob_stats['stdev']=original_prob[filt].std().to('cpu').item()
        # from scipy.stats import entropy
        # prob_dist=original_prob[filt]/self.prob_stats['mean']
        # self.prob_stats['entropy']=entropy(prob_dist.to('cpu').numpy()+1e-6)
    
    def size(self):
        return self.pts_shape[0]
    
    def compress_object(self):
        self.pts=None
        self.farthestP=None
    
    def find_pts_in_image(self, cam_params, camera_poseM, num_points=20):
        # Build a list of row/col points from a randomly sampled selection from the pointcloud
        rc_list=[]
        rr = np.arange(self.pts.shape[0])
        np.random.shuffle(rr)
        for i in range(min(num_points,self.size())):
            M2=np.matmul(cam_params.rot_matrix,camera_poseM)
            row,col=cam_params.globalXYZ_to_imageRC(self.pts[rr[i],0],self.pts[rr[i],1],self.pts[rr[i],2],M2)
            if (row>=0) and (row<cam_params.height) and (col>=0) and (col<cam_params.width):
                rc_list.append([row,col])
        return rc_list

# Based on angles alone, identify a list of images that point at the same bbox
#   This example uses a known image + bbox combination as a starting point
#   Available thresholds are:
#       dist_threshold = distance from object to centroid (requires loading depth image)
#       angular_dist   = maximum offset angle from image center
def identify_related_images_from_bbox(cam_info:camera_params, 
                            fList:rgbd_file_list, 
                            primary_image_key:int, 
                            xy_bbox:np.array): #[xmin, ymin, xmax, ymax]
    depth_fName=fList.get_depth_fileName(primary_image_key)
    depthI=cv2.imread(depth_fName,-1)
    #Calculuate median depth over area
    roi=depthI[xy_bbox[1]:xy_bbox[3],xy_bbox[0]:xy_bbox[2]]
    meanDepth=np.median(roi)/1000.0
    ctrV=np.array([((xy_bbox[0]+xy_bbox[2])/2-cam_info.cx)*meanDepth/cam_info.fx,
                   (cam_info.cy-(xy_bbox[1]+xy_bbox[3])/2)*meanDepth/cam_info.fy,
                   meanDepth,1])
    M=np.matmul(cam_info.rot_matrix,fList.get_pose(primary_image_key))
    object_pose=np.matmul(M,ctrV)
    return identify_related_images_global_pose(cam_info, fList, object_pose)     

def identify_related_images_global_pose(cam_info:camera_params, 
                            fList:rgbd_file_list, 
                            object_pose:np.array):
    # Need to load depth image
    valid_imgs=[]
    for key in fList.keys():        
        M=np.matmul(cam_info.rot_matrix,fList.get_pose(key))
        row,col=cam_info.globalXYZ_to_imageRC(object_pose[0],object_pose[1],object_pose[2],M)
        if row>=0 and row<cam_info.height and col>=0 and col<cam_info.width:
            valid_imgs.append(key)
    return valid_imgs