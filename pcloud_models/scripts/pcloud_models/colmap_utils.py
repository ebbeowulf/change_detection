import json
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from rgbd_file_list import rgbd_file_list
from change_detection.camera_params import camera_params

import pdb
# from change_detection.change_detect_common import image_comparator

## Build the camera parameter data structure from cameras.txt
def get_camera_params(colmap_dir, nerfstudio_dir):
    try:
        with open(nerfstudio_dir+"/dataparser_transforms.json","r") as fin:
            A = json.load(fin)
        cam_rot_matrix=np.identity(4)
        cam_rot_matrix[:3,:]=np.array(A['transform'])
    except Exception as e:
        print("Failed to open cameras.txt in " + colmap_dir + " - exiting")
        sys.exit(-1)

    # Load the location csv
    try:
        with open(colmap_dir+"/cameras.txt", 'r') as fin:
            A=fin.readlines()
    except Exception as e:
        print("Failed to open cameras.txt in " + colmap_dir + " - exiting")
        sys.exit(-1)


    # Remove comments at beginning
    for ln in A:
        if ln[0]=='#':
            continue
        lns=ln.split(' ')
        return camera_params(height=float(lns[3]),width=float(lns[2]),
                             fx=float(lns[4]),fy=float(lns[5]),
                             cx=float(lns[6]),cy=float(lns[7]),
                             rot_matrix=cam_rot_matrix)
                            #  rot_matrix=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]))

## Retrieve the camera poses
##      Need two files for this purpose.
##      images.txt - contains the actual image locations as
##      dataparsar_transform.json - 
def get_all_poses(colmap_dir, keyword:str):
    # Load the location csv
    try:
        with open(colmap_dir+"/images.txt", 'r') as fin:
            A=fin.readlines()
    except Exception as e:
        print("Failed to open images.txt in " + colmap_dir + " - exiting")
        os.exit(-1)

    # Remove comments at beginning
    start_line=-1
    for ln_idx,ln in enumerate(A):
        if ln[0]=='#':
            continue
        start_line=ln_idx
        break

    all_poses=dict()
    for ln_idx in range(start_line,len(A),2):
        if A[ln_idx][-1]=='\n':
            ln=A[ln_idx][:-1]
        else:
            ln=A[ln_idx]
        vals=ln.split(' ')
        if keyword is None or keyword in vals[-1]:
            quat_xyzw=[float(vals[2]),float(vals[3]),float(vals[4]),float(vals[1])]
            rot=R.from_quat(quat_xyzw) #this is an inverted matrix pointing back towards the center of the camera
            trans=np.array([float(vals[5]),float(vals[6]),float(vals[7])])
            # According to the NerfStudio documentation, pose is recovered by -rot^t * T
            #   https://colmap.github.io/format.html
            rot_matrix=rot.as_matrix()
            rot_matrix_c2w=rot_matrix.transpose()
            # This hack works - but don't use as it will probably break later
            pose=np.matmul(-rot_matrix_c2w,trans)
            all_poses[vals[-1]]={'pose': pose, 'rot_cam2world': rot_matrix.transpose()}
    return all_poses

## Build the rgbd_file_list structure for use with other pcloud utilities
def build_file_list(color_dir, depth_dir, save_dir, colmap_dir, keyword:str):
    all_poses=get_all_poses(colmap_dir,keyword=keyword)

    fList = rgbd_file_list(color_dir, depth_dir, save_dir, False)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # now re-order and generate files
    for key in all_poses.keys():
        uid=key.split('_')[-1].split('.')[0]
        number=int(uid)     
        fList.add_file(number,key,f"depth_{uid}.png")    
        rot=np.identity(4)
        rot[:3,:3]=all_poses[key]['rot_cam2world']
        rot[:3,3]=all_poses[key]['pose']        
        fList.add_pose(number,rot)

    X=np.zeros((len(fList.keys())))
    Y=np.zeros((len(fList.keys())))
    for key_id, key in enumerate(fList.keys()):
        X[key_id]=fList.get_pose(key)[0,3]
        Y[key_id]=fList.get_pose(key)[1,3]

    return fList

def build_rendered_file_list(fList_new:rgbd_file_list, rendered_image_dir,save_dir):
    # Need to build a second file list with all of the rendered images
    fList_renders=rgbd_file_list(rendered_image_dir,rendered_image_dir,save_dir,False)
    for key in fList_new.keys():
        suffix=fList_new.get_depth_fileName(key).split('/')[-1].split('_')[-1]
        fList_renders.add_file(key,"rgb_"+suffix,"depth_"+suffix)
        M=fList_new.get_pose(key)
        fList_renders.add_pose(key,M)    
    return fList_renders