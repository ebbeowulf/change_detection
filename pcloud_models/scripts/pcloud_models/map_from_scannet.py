from change_detection.yolo_segmentation import yolo_segmentation
import cv2
import numpy as np
import argparse
import glob
import pdb
import pickle
import os
import torch
import open3d as o3d
import map_utils
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def load_camera_info(info_file):
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            if key=='sceneType':
                if val[-1]=='\n':
                    val=val[:-1]
                info_dict[key] = val
            elif key=='axisAlignment':
                info_dict[key] = np.fromstring(val, sep=' ')
            else:
                info_dict[key] = float(val)

    if 'axisAlignment' not in info_dict:
       info_dict['rot_matrix'] = np.identity(4)
    else:
        info_dict['rot_matrix'] = info_dict['axisAlignment'].reshape(4, 4)
    return info_dict

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
    
def build_file_structure(input_dir):
    all_files=dict()
    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(input_dir+'/*.txt')
    for fName in txt_files:
        try:
            ppts=fName.split('.')
            rootName=ppts[0]
            number=int(ppts[0].split('-')[-1])
            all_files[number]={'root': rootName, 'poseFile': fName, 'depthFile': rootName+'.depth_reg.png', 'colorFile': rootName+'.color.jpg'}
        except Exception as e:
            continue
    return all_files

def load_all_poses(all_files:dict):
    for key in all_files.keys():
        pose=read_scannet_pose(all_files[key]['poseFile'])
        all_files[key]['pose']=pose



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_directory',type=str,help='location of raw images to process')
    parser.add_argument('param_file',type=str,help='camera parameter file for this scene')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    # parser.add_argument('--targets',type=list, nargs='+', default=None, help='Set of target classes to build point clouds for')
    args = parser.parse_args()
    all_files=build_file_structure(args.scan_directory)
    params=load_camera_info(args.param_file)
    load_all_poses(all_files)
    map_utils.process_images(all_files)
    # create_map(args.tgt_class,all_files)
    # obj_list=create_object_list(all_files)
    # obj_list=['bed','vase','potted plant','tv','refrigerator','chair']
    # create_combined_xyzrgb(obj_list, all_files, params)
    if args.targets is None:
        obj_list=map_utils.create_object_list(all_files)
        print("Detected Objects (Conf > 0.75)")
        high_conf_list=map_utils.get_high_confidence_objects(obj_list, confidence_threshold=0.75)
        print(high_conf_list)
        pclouds=map_utils.create_pclouds(high_conf_list,all_files,params,conf_threshold=0.5)
    else:
        pclouds=map_utils.create_pclouds(args.targets,all_files,params,conf_threshold=0.5)
    for key in pclouds.keys():
        fileName=args.scan_directory+"/"+key+".ply"
        pcd=map_utils.pointcloud_open3d(pclouds[key]['xyz'],pclouds[key]['rgb'])
        if 0:
            o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(fileName,pcd)

