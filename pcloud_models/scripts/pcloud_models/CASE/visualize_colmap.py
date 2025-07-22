from map_from_scannet import build_file_structure, load_camera_info
from map_utils import visualize_combined_xyzrgb, identify_related_images_from_bbox
from rgbd_file_list import rgbd_file_list
from camera_params import camera_params
import argparse
import open3d as o3d
import pdb
from colmap_utils import get_camera_params, build_file_list
import numpy as np
import os

# def read_transforms(colmap_dir):
#     import json
#     with open(colmap_dir+'/transforms.json', 'r') as fin:
#         A=json.load(fin)

#     appliedT=np.array([[1,0,0],[0,0,1],[0,-1,0]])
#     applied_scale=0.146

#     all_poses=dict()
#     for frame in A['frames']:        
#         iName=frame['file_path'].split('/')[-1]
#         tf=np.array(frame['transform_matrix'])
#         all_poses[iName]={'pose': tf[:3,3]*0.146, 'rot_cam2world': -tf[:3,:3]}
#     # pdb.set_trace()
#     return all_poses

# def build_fList(color_dir, depth_dir, save_dir, colmap_dir, keyword:str):
#     all_poses=read_transforms(colmap_dir)

#     fList = rgbd_file_list(color_dir, depth_dir, save_dir, False)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)

#     # now re-order and generate files
#     for key in all_poses.keys():
#         if keyword in key:
#             uid=key.split('_')[-1].split('.')[0]
#             number=int(uid)     
#             fList.add_file(number,key,f"depth_{uid}.png")    
#             rot=np.identity(4)
#             rot[:3,:3]=all_poses[key]['rot_cam2world']
#             rot[:3,3]=all_poses[key]['pose']        
#             fList.add_pose(number,rot)

#     X=np.zeros((len(fList.keys())))
#     Y=np.zeros((len(fList.keys())))
#     for key_id, key in enumerate(fList.keys()):
#         X[key_id]=fList.get_pose(key)[0,3]
#         Y[key_id]=fList.get_pose(key)[1,3]

#     return fList

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('root_image_dir',type=str,help='root project folder where the images and colmap info are stored')
    parser.add_argument('--color_dir',type=str,default='images_combined',help='where are the color images? (default=images_combined)')
    parser.add_argument('--depth_dir',type=str,default='renders',help='where are the depth images? (default=renders)')
    parser.add_argument('--colmap_dir',type=str,default='colmap_combined/sparse',help='where are the images + cameras.txt files? (default=colmap_combined/sparse)')
    parser.add_argument('--frame_keyword',type=str,default=None,help='a keyword to use when parsing the transforms file (default=None)')
    parser.add_argument('--save_dir',type=str,default='save_results', help='subdirectory of color_dir in which to store the intermediate files (default=save_results)')
    parser.add_argument('--headless', dest='draw', action='store_false')
    parser.add_argument('--skip',type=int,default=10,help='number of images to skip between adding frames')
    parser.add_argument('--max_image_count',type=int,default=50,help='maximum number of images to add to the point cloud')
    parser.add_argument('--max_depth',type=float,default=10.0,help='maximum depth to use when compiling the point cloud')
    args = parser.parse_args()

    save_dir=f"{args.root_image_dir}/{args.save_dir}/"
    color_image_dir=f"{args.root_image_dir}/{args.color_dir}/"
    depth_image_dir=f"{args.root_image_dir}/{args.depth_dir}/"
    colmap_dir=f"{args.root_image_dir}/{args.colmap_dir}/"
    params=get_camera_params(colmap_dir,args.nerfacto_dir)
    # params=get_camera_params(args.colmap_dir)
    fList=build_file_list(color_image_dir,depth_image_dir,save_dir,colmap_dir,args.frame_keyword)

    pcd=visualize_combined_xyzrgb(fList, params, howmany_files=args.max_image_count, skip=args.skip, max_depth=args.max_depth)
    o3d.io.write_point_cloud(fList.get_combined_pcloud_fileName(),pcd)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
    if args.draw:
        o3d.visualization.draw_geometries([cl])

