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
            if key in ['colorHeight', 'colorWidth', 'fx_color', 'fy_color', 'mx_color', 'my_color',
                       'depthHeight', 'depthWidth', 'fx_depth', 'fy_depth', 'mx_depth', 'my_depth']:
                info_dict[key] = float(val)
            elif key == 'axisAlignment':
                info_dict[key] = np.fromstring(val, sep=' ').reshape(4, 4)
    return info_dict

def process_scannet(root_dir):
    # Load camera parameters
    caminfo_file = os.path.join(root_dir, "..", "scene0050_00.txt")
    info_dict = load_camera_info(caminfo_file)
    
    # Create camera parameters object
    color_params = camera_params(
        height=info_dict['colorHeight'],
        width=info_dict['colorWidth'],
        fx=info_dict['fx_color'],
        fy=info_dict['fy_color'],
        cx=info_dict['mx_color'],
        cy=info_dict['my_color'],
        rot_matrix=info_dict['axisAlignment']
    )
    
    # Get all color images
    color_files = sorted(glob.glob(os.path.join(root_dir, "*.color.jpg")))
    
    for color_file in color_files:
        # Get the base filename
        base_name = os.path.splitext(color_file)[0]
        base_name = os.path.splitext(base_name)[0]
        # Load color and depth images
        color = cv2.imread(color_file)
        depth = cv2.imread(base_name + ".depth.pgm", -1)
        
        if color is None or depth is None:
            print(f"Could not load files for {base_name}")
            continue
        
        # Convert depth to float (in meters)
        depth = depth.astype(np.float32) / 1000.0
        
        # Create depth registration using OpenCV
        depth_reg = np.zeros((color_params.height, color_params.width), dtype=np.float32)
        
        # For each pixel in the color image, find the corresponding depth
        for y in range(color.shape[0]):
            for x in range(color.shape[1]):
                # Convert pixel coordinates to normalized coordinates
                u = (x - color_params.cx) / color_params.fx
                v = (y - color_params.cy) / color_params.fy
                
                # Get depth value
                if y < depth.shape[0] and x < depth.shape[1]:
                    z = depth[y, x]
                    
                    # Convert to 3D point
                    X = z * u
                    Y = z * v
                    Z = z
                    
                    # Project back to image coordinates
                    x_proj = int(X * color_params.fx / Z + color_params.cx)
                    y_proj = int(Y * color_params.fy / Z + color_params.cy)
                    
                    # Store depth value in registered image
                    if 0 <= x_proj < depth_reg.shape[1] and 0 <= y_proj < depth_reg.shape[0]:
                        depth_reg[y, x] = depth[y_proj, x_proj]
        
        # Save the registered depth image
        output_file = base_name + ".depth_reg.png"
        print(f"Saving registered depth to {output_file}")
        cv2.imwrite(output_file, depth_reg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ScanNet data")
    parser.add_argument('root_dir', type=str, help='Directory containing color and depth images')
    args = parser.parse_args()
    
    process_scannet(args.root_dir)
