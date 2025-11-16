import numpy as np
import argparse
import glob
import pickle
import os
from rgbd_file_list import rgbd_file_list
from camera_params import camera_params
from scannet_processing import load_camera_info, get_scene_type, read_scannet_pose, build_file_structure

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
    parser.add_argument('--draw', dest='draw', action='store_true')
    parser.set_defaults(draw=False)
    parser.add_argument('--use_connected_components', dest='use_cc', action='store_true')
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
    
    import map_utils
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
        pclouds=map_utils.create_pclouds(args.targets,fList,params, args.yolo, conf_threshold=args.threshold, use_connected_components=False)

    import open3d as o3d
    for key in pclouds.keys():
        ply_fileName=fList.get_combined_pcloud_fileName(key)
        pcd=map_utils.pointcloud_open3d(pclouds[key]['xyz'],pclouds[key]['rgb'])
        o3d.io.write_point_cloud(ply_fileName,pcd)
        raw_fileName=fList.get_combined_raw_fileName(key)

        with open(raw_fileName, 'wb') as handle:
            pickle.dump(pclouds[key], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if args.draw:
            o3d.visualization.draw_geometries([pcd])

