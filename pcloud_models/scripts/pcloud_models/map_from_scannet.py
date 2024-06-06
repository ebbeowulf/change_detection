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
# DEVICE = torch.device("cpu")

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
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('--param_file',type=str,default=None,help='camera parameter file for this scene - default is of form <raw_dir>/scene????_??.txt')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--threshold',type=float,default=0.75, help='proposed detection threshold (default = 0.75)')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    parser.add_argument('--clip', dest='yolo', action='store_false')
    parser.add_argument('--yolo', dest='yolo', action='store_true')
    parser.set_defaults(yolo=True)
    # parser.add_argument('--targets',type=list, nargs='+', default=None, help='Set of target classes to build point clouds for')
    args = parser.parse_args()
    save_dir=args.root_dir+"/"+args.save_dir
    fList=build_file_structure(args.root_dir+"/"+args.raw_dir, save_dir)
    if args.param_file is not None:
        par_file=args.param_file
    else:
        s_root=args.root_dir.split('/')
        if s_root[-1]=='':
            par_file=args.root_dir+"%s.txt"%(s_root[-2])
        else:
            par_file=args.root_dir+"/%s.txt"%(s_root[-1])
    params=load_camera_info(par_file)
    if args.yolo:
        map_utils.process_images_with_yolo(fList)
    else:
        map_utils.process_images_with_clip(fList,args.targets)
        # map_utils.clip_threshold_evaluation(fList, args.targets,args.threshold)
    if args.targets is None:
        if not args.yolo:
            print("Must specify a target if using clip segmentation")
            quit()
        obj_list=map_utils.create_yolo_object_list(fList)
        print("Detected Objects (Conf > %f)"%(args.threshold))
        high_conf_list=map_utils.get_high_confidence_objects(obj_list, confidence_threshold=args.threshold)
        print(high_conf_list)
        pclouds=map_utils.create_pclouds(high_conf_list,fList,params, args.yolo, conf_threshold=args.threshold)
    else:
        pclouds=map_utils.create_pclouds(args.targets,fList,params, args.yolo, conf_threshold=args.threshold)

    for key in pclouds.keys():
        ply_fileName=save_dir+"/"+key+".ply"
        pcd=map_utils.pointcloud_open3d(pclouds[key]['xyz'],pclouds[key]['rgb'])
        o3d.io.write_point_cloud(ply_fileName,pcd)
        raw_fileName=save_dir+"/"+key+".raw.pkl"

        with open(raw_fileName, 'wb') as handle:
            pickle.dump(pclouds[key], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if 0:
            o3d.visualization.draw_geometries([pcd])

