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
            if line[-1]=='\n':
                line=line[:-1]
            (key, val) = line.split(" = ")
            if key=='sceneType':
                info_dict[key] = val
            elif key=='axisAlignment':
                info_dict[key] = np.fromstring(val, sep=' ')
            elif key=='colorToDepthExtrinsics':
                info_dict[key] = np.fromstring(val, sep=' ')
            else:
                info_dict[key] = float(val)

    if 'axisAlignment' not in info_dict:
       rot_matrix = np.identity(4)
    else:
        rot_matrix = info_dict['axisAlignment'].reshape(4, 4)

    return map_utils.camera_params(info_dict['colorHeight'], info_dict['colorWidth'],info_dict['fx_color'],info_dict['fy_color'],info_dict['mx_color'],info_dict['my_color'],rot_matrix)

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
    
def build_file_structure(image_dir, save_dir):
    fList = map_utils.rgbd_file_list(image_dir, image_dir, save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(image_dir+'/*.txt')
    for fName in txt_files:
        try:
            ppts=fName.split('.')
            rootName=ppts[0].split('/')[-1]
            number=int(rootName.split('-')[-1])
            pose=read_scannet_pose(fName)
            fList.add_file(number,rootName+'.color.jpg',rootName+'.depth_reg.png')
            fList.add_pose(number, pose)
        except Exception as e:
            continue
    return fList

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_directory',type=str,help='location of raw images to process')
    parser.add_argument('param_file',type=str,help='camera parameter file for this scene')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    parser.add_argument('--clip', dest='yolo', action='store_false')
    parser.add_argument('--yolo', dest='yolo', action='store_true')
    parser.set_defaults(yolo=True)
    # parser.add_argument('--targets',type=list, nargs='+', default=None, help='Set of target classes to build point clouds for')
    args = parser.parse_args()
    fList=build_file_structure(args.scan_directory, args.scan_directory+"/save_results")
    params=load_camera_info(args.param_file)
    if args.yolo:
        map_utils.process_images_with_yolo(fList)
    else:
        map_utils.process_images_with_clip(fList,args.targets)
    # create_map(args.tgt_class,all_files)
    # obj_list=create_object_list(all_files)
    # obj_list=['bed','vase','potted plant','tv','refrigerator','chair']
    # create_combined_xyzrgb(obj_list, all_files, params)
    if args.targets is None:
        if not args.yolo:
            print("Must specify a target if using clip segmentation")
            quit()
        obj_list=map_utils.create_yolo_object_list(fList)
        print("Detected Objects (Conf > 0.75)")
        high_conf_list=map_utils.get_high_confidence_objects(obj_list, confidence_threshold=0.75)
        print(high_conf_list)
        pclouds=map_utils.create_pclouds(high_conf_list,fList,params, args.yolo, conf_threshold=0.5)
    else:
        pclouds=map_utils.create_pclouds(args.targets,fList,params, args.yolo, conf_threshold=0.5)
    for key in pclouds.keys():
        fileName=args.scan_directory+"/"+key+".ply"
        pcd=map_utils.pointcloud_open3d(pclouds[key]['xyz'],pclouds[key]['rgb'])
        if 0:
            o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(fileName,pcd)

