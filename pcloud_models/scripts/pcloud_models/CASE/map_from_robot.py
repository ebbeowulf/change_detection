import argparse
import map_utils
import os
from change_detection.image_set import image_set
import pdb
import numpy as np

K_ROTATED=[906.7647705078125, 0.0, 368.2167053222656, 
                0.0, 906.78173828125, 650.24609375,                 
                0.0, 0.0, 1.0]
F_X=K_ROTATED[0]
C_X=K_ROTATED[2]
F_Y=K_ROTATED[4]
C_Y=K_ROTATED[5]

def load_camera_info():
    return map_utils.camera_params(1280,720,F_X,F_Y,C_X,C_Y,np.identity(4))

def build_file_structure(root_dir,color_dir, depth_dir, save_dir, images_csv):
    save_path=root_dir+save_dir
    fList = map_utils.rgbd_file_list(root_dir+"/"+color_dir, root_dir+"/"+depth_dir, save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    IS=image_set(root_dir+images_csv)
    for is_key in IS.get_pose_list():
        depth_fName="depth_"+IS.all_images[is_key]['name'].split('_')[1]
        fList.add_file(IS.all_images[is_key]['id'],IS.all_images[is_key]['name'],depth_fName)
        fList.add_pose(IS.all_images[is_key]['id'],IS.all_images[is_key]['global_poseM'])
        
    return fList


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='should contain the color + depth directories and the associated image csv files. See change detection setup for more information')
    parser.add_argument('--color-dir',type=str,default="rotated", help='Name of directory holding the color image files. Should align with names in image_csv')
    parser.add_argument('--depth-dir',type=str,default="depth_rotated", help='Name of directory holding the depth image files. Images have same numbers as in image_csv, but use the prefix "depth_"')
    parser.add_argument('--save-dir',type=str,default="processed", help='Name of directory to store intermediate files')
    parser.add_argument('--images-csv',type=str,default="new_images.txt", help='Pose file')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    args = parser.parse_args()
    fList=build_file_structure(args.root_dir, args.color_dir, args.depth_dir,args.save_dir,args.images_csv)
    
    params=load_camera_info()
    map_utils.create_pclouds_from_images(fList,params,args.targets,display_pclouds=False)
