# Utility script to visualize a primary point cloud with an overlay from a secondary point cloud stored in a pickle file.

import argparse
import open3d as o3d
import pdb
import os
import pickle as pkl
import numpy as np
#from change_pcloud_utils.map_utils import pointcloud_open3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('primary_pcl',type=str,help='location of the primary pointcloud to visualize')
    parser.add_argument('overlay_pkl',type=str,help='location of secondary pointcloud (in pkl format) to visualize')
    parser.add_argument('--cam_rot_vector',type=str,default='-0.021964207741166447 -6.2021677080551063 0.67452825798516125 4.0262047501955758',help='Camera facing in the right direction to set the vertical so that height based thresholds can apply')
    parser.add_argument('--height_threshold',type=float,default=100000.00,help='Height threshold to apply')
    args = parser.parse_args()

    # From co-pilot
    # T = np.eye(4)
    # T[:3, :3] = R_world
    # T[:3, 3] = -R_world @ t_cam  # optional: if you want to place the cloud in world space

    # # Apply to homogeneous coordinates
    # points_h = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    # rotated_points = (T @ points_h.T).T[:, :3]


    primary=o3d.io.read_point_cloud(args.primary_pcl)
    points = np.asarray(primary.points)
    mask = points[:,1] > 1.5
    primary_pcl=o3d.geometry.PointCloud()
    primary_pcl.points = o3d.utility.Vector3dVector(points[mask]) # normals and colors are unchanged
    primary_pcl.colors = o3d.utility.Vector3dVector(np.asarray(primary.colors)[mask])
    
    o3d.visualization.draw_geometries([primary_pcl])


    # with open(args.overlay_pkl, 'rb') as handle:
    #     secondary=pkl.load(handle)
    # color=np.zeros([secondary['xyz'].shape[0],3],dtype=np.uint8)
    # color[:,2]=255
    # secondary_pcl=pointcloud_open3d(secondary['xyz'].cpu().numpy(),color)
    # # o3d.visualization.draw_geometries([secondary_pcl])
    # combo_pcl=primary+secondary_pcl
    # o3d.visualization.draw_geometries([combo_pcl])
    # # pdb.set_trace()
