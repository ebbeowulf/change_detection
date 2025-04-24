#!/usr/bin/env python3

import pickle
import numpy as np
import open3d as o3d
import cv2
import argparse

def load_point_cloud(pkl_file):
    """Load point cloud from .raw.pkl file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data['xyz'], data['rgb'], data['probs']

def save_as_ply(xyz, rgb, output_file):
    """Save point cloud as PLY file"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Saved PLY file to {output_file}")

def project_to_2d(xyz, rgb, output_file, width=640, height=480):
    """Project 3D points to 2D image plane with white background"""
    # Create a white background image
    image = 255 * np.ones((height, width, 3), dtype=np.uint8)
    
    # Get the range of the point cloud
    x_min, x_max = np.min(xyz[:,0]), np.max(xyz[:,0])
    y_min, y_max = np.min(xyz[:,1]), np.max(xyz[:,1])
    z_min, z_max = np.min(xyz[:,2]), np.max(xyz[:,2])
    
    # Calculate the center of the point cloud
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2
    
    # Calculate the scaling factor based on the largest dimension
    scale_factor = min(width, height) / max(x_max - x_min, y_max - y_min)
    
    # Project points
    for i in range(xyz.shape[0]):
        # Scale and shift coordinates to fit image
        x = int((xyz[i,0] - center_x) * scale_factor + width / 2)
        y = int((xyz[i,1] - center_y) * scale_factor + height / 2)
        
        # Check if point is within image bounds
        if 0 <= x < width and 0 <= y < height:
            image[y, x] = rgb[i]
    
    # Save the image
    cv2.imwrite(output_file, image)
    print(f"Saved 2D projection to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud from .raw.pkl file")
    parser.add_argument('input_file', type=str, help='Path to .raw.pkl file')
    parser.add_argument('--output_ply', type=str, default=None, help='Output PLY file name')
    parser.add_argument('--output_2d', type=str, default=None, help='Output 2D image file name')
    args = parser.parse_args()
    
    # Load point cloud
    xyz, rgb, probs = load_point_cloud(args.input_file)
    
    # Save as PLY if requested
    if args.output_ply:
        save_as_ply(xyz, rgb, args.output_ply)
    
    # Save 2D projection if requested
    if args.output_2d:
        project_to_2d(xyz, rgb, args.output_2d)

if __name__ == "__main__":
    main()
