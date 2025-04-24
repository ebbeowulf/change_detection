#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob
import argparse
from camera_params import camera_params

def load_camera_info(info_file):
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            (key, val) = line.split(" = ")
            if key == 'axisAlignment':
                info_dict[key] = np.fromstring(val, sep=' ').reshape(4, 4)
            elif key in ['colorHeight', 'colorWidth', 'fx_color', 'fy_color', 'mx_color', 'my_color',
                        'depthHeight', 'depthWidth', 'fx_depth', 'fy_depth', 'mx_depth', 'my_depth']:
                info_dict[key] = float(val)
    return info_dict

def register_depth(color_img, depth_img, color_params, depth_params):
    # Convert depth to float (in meters)
    depth = depth_img.astype(np.float32) / 1000.0
    
    # Create depth registration using OpenCV
    depth_reg = np.zeros((color_params.height, color_params.width), dtype=np.float32)
    
    # For each pixel in the depth image, find the corresponding color pixel
    for y in range(depth_params.height):
        for x in range(depth_params.width):
            # Convert depth pixel coordinates to normalized coordinates
            u = (x - depth_params.cx) / depth_params.fx
            v = (y - depth_params.cy) / depth_params.fy
            
            # Get depth value
            z = depth[y, x]
            
            if z > 0:  # Only process valid depth values
                # Convert to 3D point
                X = z * u
                Y = z * v
                Z = z
                
                # Project back to color image coordinates
                x_proj = int(X * color_params.fx / Z + color_params.cx)
                y_proj = int(Y * color_params.fy / Z + color_params.cy)
                
                # Store depth value in registered image
                if 0 <= x_proj < depth_reg.shape[1] and 0 <= y_proj < depth_reg.shape[0]:
                    depth_reg[y_proj, x_proj] = z
    
    return depth_reg

def process_scannet(root_dir):
    # Load camera parameters
    caminfo_file = os.path.join(root_dir, "..", "scene0050_00.txt")
    info_dict = load_camera_info(caminfo_file)
    
    # Create camera parameters objects for both color and depth
    color_params = camera_params(
        height=info_dict['colorHeight'],
        width=info_dict['colorWidth'],
        fx=info_dict['fx_color'],
        fy=info_dict['fy_color'],
        cx=info_dict['mx_color'],
        cy=info_dict['my_color'],
        rot_matrix=info_dict['axisAlignment']
    )
    
    depth_params = camera_params(
        height=info_dict['depthHeight'],
        width=info_dict['depthWidth'],
        fx=info_dict['fx_depth'],
        fy=info_dict['fy_depth'],
        cx=info_dict['mx_depth'],
        cy=info_dict['my_depth'],
        rot_matrix=info_dict['axisAlignment']
    )
    
    # Get all color images
    color_files = sorted(glob.glob(os.path.join(root_dir, "*.color.jpg")))
    
    total_files = len(color_files)
    print(f"Found {total_files} color images to process")
    
    for idx, color_file in enumerate(color_files, 1):
        # Get the base filename
        base_name = os.path.splitext(color_file)[0]
        
        # Load color and depth images
        color = cv2.imread(color_file)
        depth = cv2.imread(base_name + ".depth.pgm", -1)
        
        if color is None:
            print(f"Error: Could not load color image {color_file}")
            continue
            
        if depth is None:
            print(f"Error: Could not load depth image {base_name}.depth.pgm")
            continue
        
        print(f"Processing image {idx}/{total_files}: {base_name}")
        
        # Register depth with color
        try:
            depth_reg = register_depth(color, depth, color_params, depth_params)
            
            # Save the registered depth image
            output_file = base_name + ".depth_reg.png"
            print(f"Saving registered depth to {output_file}")
            cv2.imwrite(output_file, depth_reg * 1000)  # Convert back to millimeters
            
        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ScanNet data")
    parser.add_argument('root_dir', type=str, help='Directory containing color and depth images')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Starting processing in directory: {args.root_dir}")
    
    process_scannet(args.root_dir)
