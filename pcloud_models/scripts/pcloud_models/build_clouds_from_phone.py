import argparse
from colmap_utils import get_camera_params, build_file_list, build_rendered_file_list
from map_utils import visualize_combined_xyzrgb, identify_related_images_from_bbox, DEVICE, pointcloud_open3d, get_distinct_clusters, object_pcloud
import os
import sys
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'change_detection', 'scripts', 'change_detection'))
sys.path.append(scripts_path)
from pcloud_creation_utils import pcloud_change, pcloud_openVocab
from camera_params import camera_params
from rgbd_file_list import rgbd_file_list
import pdb
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import open3d as o3d

ABSOLUTE_MIN_CLUSTER_SIZE=500

def build_change_clouds(params:camera_params, 
                     fList_new:rgbd_file_list,
                     fList_renders:rgbd_file_list,
                     prompts:list,
                     det_threshold:float):
    pcloud=dict()
    for query in prompts:
        pcloud[query]={'xyz': torch.zeros((0,3),dtype=float,device=DEVICE), 
                       'probs': torch.zeros((0),dtype=float,device=DEVICE), 
                       'rgb': torch.zeros((0,3),dtype=float,device=DEVICE)}

    pcloud_creator=pcloud_change(params)
    for key in fList_new.keys():
        try:
            colorI_new=Image.open(fList_new.get_color_fileName(key))
            colorI_rendered=Image.open(fList_renders.get_color_fileName(key))
            depthI=cv2.imread(fList_new.get_depth_fileName(key),-1)
            M=fList_new.get_pose(key)
        except Exception as e:
            print(f"Could not load files associated with key={key}")
            continue
        
        pcloud_creator.load_image(colorI_new, depthI, M, str(key))
        results=pcloud_creator.multi_prompt_change_process(colorI_rendered, prompts, det_threshold)
        for query in prompts:
            if query in results and results[query]['xyz'].shape[0]>0:
                pcloud[query]['xyz']=torch.vstack((pcloud[query]['xyz'],results[query]['xyz']))
                pcloud[query]['probs']=torch.hstack((pcloud[query]['probs'],results[query]['probs']))
                pcloud[query]['rgb']=torch.vstack((pcloud[query]['rgb'],results[query]['rgb']))
        
    return pcloud

def build_openVocab_clouds(params:camera_params, 
                     fList_new:rgbd_file_list,
                     prompts:list,
                     det_threshold:float):
    pcloud=dict()
    for query in prompts:
        pcloud[query]={'xyz': torch.zeros((0,3),dtype=float,device=DEVICE), 
                       'probs': torch.zeros((0),dtype=float,device=DEVICE), 
                       'rgb': torch.zeros((0,3),dtype=float,device=DEVICE)}

    pcloud_creator=pcloud_openVocab(params)
    for key in fList_new.keys():
        try:
            colorI_new=Image.open(fList_new.get_color_fileName(key))
            depthI=cv2.imread(fList_new.get_depth_fileName(key),-1)
            M=fList_new.get_pose(key)
        except Exception as e:
            print(f"Could not load files associated with key={key}")
            continue
        
        pcloud_creator.load_image(colorI_new, depthI, M, str(key))
        results=pcloud_creator.multi_prompt_process(prompts, det_threshold)
        for query in prompts:
            if query in results and results[query]['xyz'].shape[0]>0:
                pcloud[query]['xyz']=torch.vstack((pcloud[query]['xyz'],results[query]['xyz']))
                pcloud[query]['probs']=torch.hstack((pcloud[query]['probs'],results[query]['probs']))
                pcloud[query]['rgb']=torch.vstack((pcloud[query]['rgb'],results[query]['rgb']))
        
    return pcloud

def save_pcloud_dict(pcloud_dict:str, save_directory:str, suffix:str):
    import pickle
    for key in pcloud_dict:
        P1=key.replace(' ','_')
        save_fName=f"{save_directory}/{P1}.{suffix}.pkl"
        with open(save_fName,'wb') as handle:
            pickle.dump(pcloud_dict[key], handle, protocol=pickle.HIGHEST_PROTOCOL)    

# Performs a single pass on merging clusters
#   will probably want to run more than once, or until the list size stops changing
def merge_clusters(cluster_list:list, merge_dist:float):
    merged_clusters=[]
    isFound=np.zeros((len(cluster_list)),dtype=bool)
    # Need to sample the cloud first
    for cluster in cluster_list:
        if cluster.farthestP is None:
            cluster.sample_pcloud(100)
    # Now step through one cluster at a time
    #   if not already match, compare it to other clusters in the list
    #   any clusters matched are marked as "found" and ignored for the 
    #   remainder of the loop
    for cl_idx, cluster in enumerate(cluster_list):

        exportCL=cluster

        #Skip if marked as found already
        if isFound[cl_idx]:
            continue

        #Else go through remainder of list and merge with close clusters
        for cl_idx2, cluster2 in enumerate(cluster_list[(cl_idx+1):]):

            if not isFound[cl_idx2] and cluster.compute_cloud_distance(cluster2)<merge_dist:
                isFound[cl_idx2]=True
                exportCL=object_pcloud(np.vstack((exportCL.pts,cluster2.pts)),num_samples=100,sample=True)
        merged_clusters.append(exportCL)
    return merged_clusters
     

def create_and_merge_clusters(pcloud_xyz:np.ndarray, 
                        gridcell_size:float):
    pcd=o3d.geometry.PointCloud()    
    F2=np.where(np.isnan(pcloud_xyz).sum(1)==0)
    xyzF2=pcloud_xyz[F2]        
    pcd.points=o3d.utility.Vector3dVector(xyzF2)
    dbscan_eps=2.4*gridcell_size

    minV=xyzF2[F2].min(0)
    object_clusters=get_distinct_clusters(pcd, 
                                    floor_threshold=minV[2],
                                    cluster_min_count=ABSOLUTE_MIN_CLUSTER_SIZE,
                                    gridcell_size=gridcell_size,
                                    eps=dbscan_eps)  
    
    # Merge clusters that are really close together
    list_count=10000
    while len(object_clusters)<list_count:
        m_clusters=merge_clusters(object_clusters, 10*gridcell_size)
        object_clusters=m_clusters
        list_count=len(object_clusters)
         
    return object_clusters

def setup_change_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('root_dir',type=str,help='root project folder where the images and colmap info are stored')
    parser.add_argument('--color_dir',type=str,default='images_combined',help='where are the color images? (default=images_combined)')
    parser.add_argument('--renders_dir',type=str,default='renders',help='where are the rendered images? (default=renders)')
    parser.add_argument('--colmap_dir',type=str,default='colmap_combined/sparse',help='where are the images + cameras.txt files? (default=colmap_combined/sparse)')
    parser.add_argument('--frame_keyword',type=str,default="new",help='a keyword to use when parsing the transforms file (default=new)')
    parser.add_argument('--save_dir',type=str,default='save_results', help='subdirectory of root_dir in which to store the intermediate files (default=save_results)')
    parser.add_argument('--queries', type=str, nargs='*', default=["General clutter", "Small items on surfaces", "Floor-level objects", "Decorative and functional items", "Trash items"],
                help='Set of target queries to build point clouds for - default is [General clutter, Small items on surfaces, Floor-level objects, Decorative and functional items, Trash items]')
    parser.add_argument('--threshold',type=float, default=0.3, help="fixed threshold to apply for change detection (default=0.1)")
    parser.add_argument('--max_route_dist',type=float,default=5.2,help='assuming a fixed route, determine a rough scale for the resulting cloud using a known distance between end points (default = 5.2)')
    parser.add_argument('--no_change', dest='use_change', action='store_false')
    parser.set_defaults(use_change=True)
    args = parser.parse_args()

    save_dir=f"{args.root_dir}/{args.save_dir}/"
    color_image_dir=f"{args.root_dir}/{args.color_dir}/"
    rendered_image_dir=f"{args.root_dir}/{args.renders_dir}/"
    colmap_dir=f"{args.root_dir}/{args.colmap_dir}/"
    params=get_camera_params(colmap_dir,args.nerfacto_dir)
    fList_new=build_file_list(color_image_dir,rendered_image_dir,save_dir,colmap_dir,args.frame_keyword)
    if args.use_change:
        fList_renders=build_rendered_file_list(fList_new, rendered_image_dir,save_dir)
    else:
        fList_renders=None
    allP=np.array([ fList_new.get_pose(key)[:3,3] for key in fList_new.keys()])
    first_pose=np.median(allP[:5,:],0)
    last_pose=np.median(allP[-5:,:],0)
    reconstructed_dist=np.sqrt(np.sum((last_pose-first_pose)**2))
    scale=args.max_route_dist/reconstructed_dist

    prompts = [ s.lower() for s in args.queries ]
    return {'params': params,
            'fList_new': fList_new,
            'fList_renders': fList_renders,
            'detection_threshold': args.threshold,
            'prompts': prompts,
            'scale': scale}

if __name__ == '__main__':
    exp_params=setup_change_experiment()

    if exp_params['fList_renders'] is not None:
        # Do the original change detection experiment
        pcloud=build_change_clouds(exp_params['params'], 
                                exp_params['fList_new'], 
                                exp_params['fList_renders'], 
                                exp_params['prompts'], 
                                exp_params['detection_threshold'])

        save_pcloud_dict(pcloud,
                        exp_params['fList_new'].intermediate_save_dir,
                        f"{exp_params['detection_threshold']}.pcloud")
    else:
        # Use open vocabulary models only - no change applied
        pcloud=build_openVocab_clouds(exp_params['params'], 
                                exp_params['fList_new'], 
                                exp_params['prompts'], 
                                exp_params['detection_threshold'])

        save_pcloud_dict(pcloud,
                        exp_params['fList_new'].intermediate_save_dir,
                        f"{exp_params['detection_threshold']}.OV.pcloud")

    # Build the o3d cloud to visualize
    # pcd_all=dict()
    # for query in prompts:
    #     if pcloud[query]['xyz'].shape[0]>1000:
    #         # pcd_all[query]=pointcloud_open3d(pcloud[query]['xyz'].cpu().numpy())
          

    # import open3d as o3d
    # combo_pcd=visualize_combined_xyzrgb(fList_new, params, skip=5)
    # save_pcloud_dict(pcloud,save_dir,f"{args.threshold}.pcloud")
    # save_pcloud_dict(pcd_all,save_dir,"0.5.pcd")
    # save_pcloud_dict({'combined':combo_pcd},save_dir,"pcd")

    # mat1 = o3d.visualization.rendering.MaterialRecord()
    # mat1.shader = 'defaultLitTransparency'
    # mat1.base_color = [1.0, 0.0, 0.0, 0.5]  # RGBA, adjust A for transparency
    # mat1.point_size = 20.0

    # o3d.visualization.draw([{'name': 'combined', 'geometry': combo_pcd, 'material': mat1},
    #                         {'name': 'trash', 'geometry':pcd_all['trash items']}],bg_color=(0.1, 0.1, 0.1, 1.0))
    # mat1 = o3d.visualization.rendering.MaterialRecord()
    # mat1.shader = 'defaultLitTransparency'
    # mat1.base_color = [1.0, 0.0, 0.0, 0.5]  # RGBA, adjust A for transparency
    # mat1.point_size = 20.0

    # mat1 = o3d.visualization.rendering.MaterialRecord()
    # mat1.shader = 'defaultLitTransparency'
    # mat1.base_color = [1.0, 0.0, 0.0, 0.5]  # RGBA, adjust A for transparency
    # mat1.point_size = 20.0
    #Build the combined pcloud for comparison



