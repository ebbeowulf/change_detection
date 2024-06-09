from map_from_scannet import build_file_structure, load_camera_info
from map_utils import visualize_combined_xyzrgb
import argparse
import open3d as o3d
import pdb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('--param_file',type=str,default=None,help='camera parameter file for this scene - default is of form <raw_dir>/scene????_??.txt')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
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

    pcd=visualize_combined_xyzrgb(fList, params, howmany_files=200, skip=5)
    o3d.io.write_point_cloud(fList.get_combined_pcloud_fileName(),pcd)
    o3d.visualization.draw_geometries([pcd])
