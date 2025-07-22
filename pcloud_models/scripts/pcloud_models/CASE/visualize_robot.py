from map_from_robot import build_file_structure, load_camera_info
from map_utils import visualize_combined_xyzrgb
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='should contain the color + depth directories and the associated image csv files. See change detection setup for more information')
    parser.add_argument('--color-dir',type=str,default="rotated", help='Name of directory holding the color image files. Should align with names in image_csv')
    parser.add_argument('--depth-dir',type=str,default="depth_rotated", help='Name of directory holding the depth image files. Images have same numbers as in image_csv, but use the prefix "depth_"')
    parser.add_argument('--save-dir',type=str,default="processed", help='Name of directory to store intermediate files')
    parser.add_argument('--images-csv',type=str,default="new_images.txt", help='Pose file')
    args = parser.parse_args()
    fList=build_file_structure(args.root_dir, args.color_dir, args.depth_dir,args.save_dir,args.images_csv)    
    params=load_camera_info()
    # map_utils.create_pclouds_from_images(fList,params,args.targets,display_pclouds=False)

    visualize_combined_xyzrgb(fList, params, howmany_files=200, skip=5)