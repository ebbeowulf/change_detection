import numpy as np
import open3d as o3d
from pcloud_creation_utils import pcloud_change
from camera_params import camera_params
from rgbd_file_list import rgbd_file_list
from find_changes import setup_change_experiment, ABSOLUTE_MIN_CLUSTER_SIZE, get_change_clusters
from map_utils import identify_related_images_global_pose
import pickle
import os
import pdb
import cv2
import matplotlib.pyplot as plt

fileName="/data2/datasets/living_room/specAI2/6_24_v1/save_results/general_clutter.0.3.pcloud.pkl"

def draw_target_circle(image, target_xyz, cam_pose, cam_info:camera_params):
    M=np.matmul(cam_info.rot_matrix,cam_pose)
    row,col=cam_info.globalXYZ_to_imageRC(target_xyz[0],target_xyz[1],target_xyz[2],M)
    if row<0 or row>=image.shape[0] or col<0 or col>=image.shape[1]:
        print("Target not in image - skipping")
        return image
    radius=int(image.shape[0]/100)
    return cv2.circle(image, (int(col),int(row)), radius=radius, color=(0,0,255), thickness=-1)

def build_change_cluster_images(exp_params, fileName):
    fileName=exp_params['fList_new']
    try:
        with open(fileName, 'rb') as handle:
            pcloud=pickle.load(handle)
    except Exception as e:
        print("pcloud file not found")
        os.exit()

    # Rescale everything ... 
    export=dict()
    if pcloud['xyz'].shape[0]>ABSOLUTE_MIN_CLUSTER_SIZE:
        clusters=get_change_clusters(pcloud['xyz'].cpu().numpy(), 0.01/exp_params['scale'])
        for cluster_idx, cluster in enumerate(clusters):
            rel_imgs=identify_related_images_global_pose(exp_params['params'],exp_params['fList_new'],cluster.centroid,None,0.5)
            for key in rel_imgs:
                iKey=int(key)
                fName=exp_params['fList_new'].get_color_fileName(iKey)
                M=exp_params['fList_new'].get_pose(iKey)
                colorI=cv2.imread(fName)
                # color_dot=draw_target_circle(colorI, cluster.centroid, M, exp_params['params'])                
                
                rc_list=[]
                rr = np.arange(cluster.pts.shape[0])
                np.random.shuffle(rr)
                for i in range(20):
                    # color_dot=draw_target_circle(color_dot, cluster.pts[rr[i],:], M, exp_params['params'])                    
                    M2=np.matmul(exp_params['params'].rot_matrix,M)
                    row,col=exp_params['params'].globalXYZ_to_imageRC(cluster.pts[rr[i],0],cluster.pts[rr[i],1],cluster.pts[rr[i],2],M2)
                    if (row>=0) and (row<exp_params['params'].height) and (col>=0) and (col<exp_params['params'].width):
                        rc_list.append([row,col])
                if len(rc_list)>3:
                    rc_list=np.array(rc_list)
                    # Expand bbox dimensions by 1.5
                    dims=(1.5*(rc_list.max(0)-rc_list.min(0)))
                    start_RC=(rc_list.mean(0)-dims/2).astype(int)
                    end_RC=start_RC+dims.astype(int)
                    # Do not export image if box extends beyond edge -
                    #   likely incomprehensible to LLM
                    if start_RC.min()<0:
                        continue
                    if end_RC[0]>exp_params['params'].height or end_RC[1]>exp_params['params'].width:
                        continue
                    color_rect=cv2.rectangle(colorI, (start_RC[1],start_RC[0]), (end_RC[1],end_RC[0]), (0,0,255), 5)

                    fName_out=exp_params['fList_new'].intermediate_save_dir+f"/cluster{cluster_idx}_{key}.png"
                    cv2.imwrite(fName_out,color_rect)

if __name__ == '__main__':
    exp_params=setup_change_experiment()    

    for key in exp_params['params'].queries:
        P1=key.replace(' ','_')
        save_directory=exp_params['fList_new'].intermediate_save_dir
        suffix=f"{exp_params['detection_threshold']}.pcloud"
        pcloud_fileName=f"{save_directory}/{P1}.{suffix}.pkl"

        image_list=build_change_cluster_images(exp_params, fileName)

