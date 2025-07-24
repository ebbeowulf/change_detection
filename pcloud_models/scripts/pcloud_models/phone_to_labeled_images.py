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
from map_utils import identify_related_images_global_pose
import pickle

ABSOLUTE_MIN_CLUSTER_SIZE=500
DEPTH_BLUR_THRESHOLD=50 #Applied to Depth images
COLOR_BLUR_THRESHOLD=5 #Applied to Color images

def build_change_clouds(params:camera_params, 
                     fList_new:rgbd_file_list,
                     fList_renders:rgbd_file_list,
                     prompts:list,
                     det_threshold:float):
    pcloud=dict()
    for query in prompts:
        pcloud[query]={'xyz': torch.zeros((0,3),dtype=float,device=DEVICE), 
                       'probs': torch.zeros((0),dtype=float,device=DEVICE), 
                       'rgb': torch.zeros((0,3),dtype=float,device=DEVICE),
                       'bboxes': dict()}

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
        
        print(fList_new.get_color_fileName(key))
        pcloud_creator.load_image(colorI_new, depthI, M, str(key),color_blur_threshold=COLOR_BLUR_THRESHOLD, depth_blur_threshold=DEPTH_BLUR_THRESHOLD)
        results, bboxes=pcloud_creator.multi_prompt_change_process(colorI_rendered, prompts, det_threshold,est_bboxes=True)
        # Instead of merging cloud here, keep it attached to the original image - so that we can draw boxes later
        for query in prompts:
            if query in results and results[query]['xyz'].shape[0]>0:
                pcloud[query]['xyz']=torch.vstack((pcloud[query]['xyz'],results[query]['xyz']))
                pcloud[query]['probs']=torch.hstack((pcloud[query]['probs'],results[query]['probs']))
                pcloud[query]['rgb']=torch.vstack((pcloud[query]['rgb'],results[query]['rgb']))
            if query in bboxes:
                pcloud[query]['bboxes'][fList_new.get_color_fileName(key)]=bboxes[query]
        
    return pcloud

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
    parser.add_argument('--no-change', dest='use_change', action='store_false')
    parser.set_defaults(use_change=True)
    parser.add_argument('--rebuild-pcloud', dest='new_pcloud', action='store_true')
    parser.set_defaults(new_pcloud=False)
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
            'scale': scale,
            'rebuild_pcloud': args.new_pcloud}

def count_points_in_box(rc_points,bbox):
    #bbox is [x_min, y_min, x_max, y_max] - so need to reverse to handle row/col
    mask=(rc_points[:,0]>bbox[1])*(rc_points[:,0]<bbox[3])*(rc_points[:,1]>bbox[0])*(rc_points[:,1]<bbox[2])
    return mask.sum()

def truncate_point(pointXY, maxX, maxY):
    return np.array([int(max(0,min(maxX,pointXY[0]))),int(max(0,min(maxY,pointXY[1])))])

def expand_bbox(bbox,multiplier,maxX,maxY):
    center=(bbox[2:]+bbox[:2])/2.0
    half_dims=multiplier*(bbox[2:]-bbox[:2])/2.0
    start_XY=center-half_dims
    end_XY=center+half_dims
    return np.hstack((truncate_point(start_XY,maxX,maxY),truncate_point(end_XY,maxX,maxY))) 

def build_change_cluster_images(exp_params, pcloud_fileName, prompt):
    try:
        with open(pcloud_fileName, 'rb') as handle:
            pcloud=pickle.load(handle)
    except Exception as e:
        print(f"pcloud file {pcloud_fileName} not found")
        os._exit(-1)

    file_prefix=prompt.replace(' ','_')
    # Rescale everything ... 
    export=dict()
    from ultralytics import SAM
    sam_model = SAM('sam2.1_l.pt')  # load an official model      
    if pcloud['xyz'].shape[0]>ABSOLUTE_MIN_CLUSTER_SIZE:
        clusters=create_and_merge_clusters(pcloud['xyz'].cpu().numpy(), 0.01/exp_params['scale'])
        for cluster_idx, cluster in enumerate(clusters):
            rel_imgs=identify_related_images_global_pose(exp_params['params'],exp_params['fList_new'],cluster.centroid,None,0.5)
            for key in rel_imgs:
                iKey=int(key)
                fName=exp_params['fList_new'].get_color_fileName(iKey)

                # We only want images that had significant change identified
                #   so that we can build tight bounding boxes around the object of interest
                if fName not in pcloud['bboxes'] or len(pcloud['bboxes'][fName])==0:
                    continue

                boxes=pcloud['bboxes'][fName]            
                M=exp_params['fList_new'].get_pose(iKey)
                sampled_points = np.array(cluster.find_pts_in_image(exp_params['params'],M,num_points=100))
                box_count=[ count_points_in_box(sampled_points, box[1]) for box in boxes]
                if max(box_count)>0:
                    whichBox=np.argmax(box_count)
                    tgt_box=np.array(boxes[whichBox][1])
                    colorI=cv2.imread(fName)
                    # Expand bbox dimensions by 1.5
                    expand_bbox(tgt_box, 1.5, exp_params['params'].width, exp_params['params'].height)
                    color_rect=cv2.rectangle(colorI, tgt_box[:2], tgt_box[2:], (0,0,255), 5)
                    pdb.set_trace()
                    sam_results = sam_model(colorI, bboxes=tgt_box)
                    cv2.imshow("image",color_rect)
                    cv2.waitKey(0)
                #     rc_list=np.array(rc_list)
                #     # Expand bbox dimensions by 1.5
                #     dims=(1.5*(rc_list.max(0)-rc_list.min(0)))
                #     start_RC=(rc_list.mean(0)-dims/2).astype(int)
                #     end_RC=start_RC+dims.astype(int)
                #     # Do not export image if box extends beyond edge -
                #     #   likely incomprehensible to LLM
                #     if start_RC.min()<0:
                #         continue
                #     if end_RC[0]>exp_params['params'].height or end_RC[1]>exp_params['params'].width:
                #         continue
                #     color_rect=cv2.rectangle(colorI, (start_RC[1],start_RC[0]), (end_RC[1],end_RC[0]), (0,0,255), 5)

                #     if exp_params['fList_renders'] is not None:
                #         fName_out=exp_params['fList_new'].intermediate_save_dir+f"/{file_prefix}_{cluster_idx}_{key}.png"
                #     else:
                #         fName_out=exp_params['fList_new'].intermediate_save_dir+f"/{file_prefix}_{cluster_idx}_{key}.OV.png"

                #     cv2.imwrite(fName_out,color_rect)

def build_pclouds(exp_params):
    # build clouds if necessary - return list of filenames for saved pclouds
    pcloud_fNames=dict()
    all_files_exist=True
    for key in exp_params['prompts']:
        P1=key.replace(' ','_')
        if exp_params['fList_renders'] is not None:
            pcloud_fNames[key]=f"{exp_params['fList_new'].intermediate_save_dir}/{P1}.{exp_params['detection_threshold']}.pcloud.pkl"
        else:
            pcloud_fNames[key]=f"{exp_params['fList_new'].intermediate_save_dir}/{P1}.{exp_params['detection_threshold']}.OV.pcloud.pkl"
        
        # Does the file exist already?
        if not os.path.exists(pcloud_fNames[key]):
            all_files_exist=False

    # Rebuild pclouds if requested 
    if not all_files_exist or exp_params['rebuild_pcloud']:
        if exp_params['fList_renders'] is not None:
            # Do the original change detection experiment
            pcloud=build_change_clouds(exp_params['params'], 
                                    exp_params['fList_new'], 
                                    exp_params['fList_renders'], 
                                    exp_params['prompts'], 
                                    exp_params['detection_threshold'])            
        else:
            # Use open vocabulary models only - no change applied
            pcloud=build_openVocab_clouds(exp_params['params'], 
                                    exp_params['fList_new'], 
                                    exp_params['prompts'], 
                                    exp_params['detection_threshold'])
        # Save the result
        for key in pcloud:
            with open(pcloud_fNames[key],'wb') as handle:
                pickle.dump(pcloud[key], handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
    # Return the list of files to be loaded
    return pcloud_fNames

if __name__ == '__main__':
    exp_params=setup_change_experiment()

    # We build the point clouds, but save to disk in order to 
    #   save memory for clustering
    pcloud_fNames=build_pclouds(exp_params)

    for key in pcloud_fNames.keys():
        image_list=build_change_cluster_images(exp_params, pcloud_fNames[key], key)


