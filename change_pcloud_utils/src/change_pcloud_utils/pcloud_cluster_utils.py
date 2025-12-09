# Utility functions for clustering point clouds.
# Includes:
#   - merge_clusters: merge nearby clusters based on distance
#   - merge_by_bounding_box: merge clusters based on bounding box overlap in images
#   - create_and_merge_clusters: create clusters from point cloud and merge nearby ones
#   - build_change_cluster_images: generate labeled images for each cluster of change
#   - count_points_in_box: count points within a bounding box
#   - truncate_point: truncate point coordinates to be within image bounds
#   - expand_bbox: expand a bounding box by a multiplier

from change_pcloud_utils.map_utils import identify_related_images_global_pose, get_weighted_clusters, object_pcloud
import numpy as np
import open3d as o3d
from change_pcloud_utils.rgbd_file_list import rgbd_file_list
from change_pcloud_utils.camera_params import camera_params
import torch

ABSOLUTE_MIN_CLUSTER_SIZE=100
FARTHESTP_SAMPLE_SIZE=100
MERGE_CLUSTER_MULTIPLIER=-1 #20
DBSCAN_EPS_MULTIPLIER=5

# Performs a single pass on merging clusters
#   will probably want to run more than once, or until the list size stops changing
def merge_clusters(cluster_list:list, merge_dist:float):
    merged_clusters=[]
    isFound=np.zeros((len(cluster_list)),dtype=bool)
    # Need to sample the cloud first
    for cluster in cluster_list:
        if cluster.farthestP is None:
            cluster.sample_pcloud(FARTHESTP_SAMPLE_SIZE)
    
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
                exportCL=object_pcloud(np.vstack((exportCL.pts,cluster2.pts)),num_samples=FARTHESTP_SAMPLE_SIZE,sample=True)
        merged_clusters.append(exportCL)
    return merged_clusters

def merge_by_bounding_box(clusters, 
                          pcloud,
                          fList_new:rgbd_file_list,
                          params:camera_params,
                          overlap_threshold=0.0 # log-likelihood threshold (0=50%)
                          ):
    from shapely import Polygon
    cluster_image_dict=dict()

    for cluster_idx1, cluster in enumerate(clusters):
        # Get the set of images where the cluster centroid is visible
        rel_imgs=identify_related_images_global_pose(params,fList_new,cluster.centroid)
        cluster_image_dict[cluster_idx1]=dict()
        for key in rel_imgs:
            iKey=int(key)
            fName=fList_new.get_color_fileName(iKey)

            # We only want images that had significant change identified
            #   so that we can build tight bounding boxes around the object of interest
            if fName not in pcloud['bboxes'] or len(pcloud['bboxes'][fName])==0:
                continue

            boxes=pcloud['bboxes'][fName]            
            M=fList_new.get_pose(iKey)
            sampled_points = np.array(cluster.find_pts_in_image(params,M,num_points=FARTHESTP_SAMPLE_SIZE))

            if len(sampled_points)<ABSOLUTE_MIN_CLUSTER_SIZE:
                # skip this image - too few points align
                continue
                
            # Determine which of the saved boxes most closely aligns with this cluster
            box_count=[ count_points_in_box(sampled_points, box[1]) for box in boxes]

            prior=max(box_count)/sampled_points.shape[0]
            if max(box_count)==0:
                # skip this image - no points align
                continue

            whichBox=np.argmax(box_count)
            tgt_box=np.array(boxes[whichBox][1])
            
            poly=Polygon(np.array([[tgt_box[0],tgt_box[1]],
                                    [tgt_box[0],tgt_box[3]],
                                    [tgt_box[2],tgt_box[3]],
                                    [tgt_box[2],tgt_box[1]],
                                    [tgt_box[0],tgt_box[1]]]))
            
            cluster_image_dict[cluster_idx1][iKey]=[fName,poly,prior]         
    
    # At this point, we have a list of boxes/polygons per image that correspond to each cluster
    #   Now we will apply a probababilistic scoring function to match boxes that have a significant overlap
    scoring=dict()
    for cluster_idx1 in cluster_image_dict.keys():

        scoring[cluster_idx1]=dict()

        for cluster_idx2 in cluster_image_dict.keys():

            if cluster_idx1==cluster_idx2:
                continue
            # Keep track of cumulative evidence
            scoring[cluster_idx1][cluster_idx2]=0

            # Score each image and add to the cumulative
            for image_key1 in cluster_image_dict[cluster_idx1]:
                IOU=0
                P1=cluster_image_dict[cluster_idx1][image_key1][2]
                P2=0
                if image_key1 in cluster_image_dict[cluster_idx2]:
                    P2=cluster_image_dict[cluster_idx2][image_key1][2]
                    # Calculate IOU if the two boxes intersect
                    if cluster_image_dict[cluster_idx1][image_key1][1].intersects(cluster_image_dict[cluster_idx2][image_key1][1]):
                        isect=cluster_image_dict[cluster_idx1][image_key1][1].intersection(cluster_image_dict[cluster_idx2][image_key1][1]).area
                        IOU=isect/(cluster_image_dict[cluster_idx1][image_key1][1].area + cluster_image_dict[cluster_idx2][image_key1][1].area - isect)
                IOU=min(0.95,max(0.05,IOU))
                P1=min(0.99,max(0.01,P1))
                P2=min(0.99,max(0.01,P2))

                # Update using log odds                
                scoring[cluster_idx1][cluster_idx2]+=(np.log(IOU)-np.log(1-IOU))
                scoring[cluster_idx1][cluster_idx2]+=(np.log(P1)-np.log(1-P1))
                scoring[cluster_idx1][cluster_idx2]+=(np.log(P2)-np.log(1-P2))

    # Merge clusters based on scoring > threshold - need to make sure scoring agrees in both directions
    #   before actually applying the merge
    merged_clusters=[]
    isMerged={ key: False for key in cluster_image_dict.keys() }
    #Else go through remainder of list and merge with close clusters
    for cluster_idx1 in scoring.keys():
        # skip clusters we have already merged
        if isMerged[cluster_idx1]:
            continue

        exportCL=clusters[cluster_idx1]

        for merge_candidate in scoring[cluster_idx1].keys():
            if isMerged[merge_candidate]:
                continue

            # Is the scoring > 0.5 in both directions?
            if  scoring[cluster_idx1][merge_candidate]>overlap_threshold and scoring[merge_candidate][cluster_idx1]>overlap_threshold:
                isMerged[merge_candidate]=True
                PS1=exportCL.prob_stats
                PS2=clusters[merge_candidate].prob_stats
                exportCL=object_pcloud(np.vstack((exportCL.pts,clusters[merge_candidate].pts)),
                                       num_samples=FARTHESTP_SAMPLE_SIZE,
                                       sample=False)
                # Need to update the stats...
                exportCL.prob_stats=dict()
                exportCL.prob_stats['max']=max(PS1['max'],PS2['max'])
                exportCL.prob_stats['pcount']=(PS1['pcount']+PS2['pcount'])
                exportCL.prob_stats['mean']=(PS1['mean']*PS1['pcount'] + PS2['mean']*PS2['pcount'])/exportCL.prob_stats['pcount']
                exportCL.prob_stats['prob_sum']=PS1['prob_sum']+PS2['prob_sum']
                exportCL.prob_stats['stdev']=np.sqrt(PS1['stdev']*PS1['stdev'] + PS2['stdev']*PS2['stdev'])

        exportCL.sample_pcloud(FARTHESTP_SAMPLE_SIZE)
        merged_clusters.append(exportCL)
        isMerged[cluster_idx1]=True

    return merged_clusters

# def create_and_merge_clusters(pcloud_xyz:np.ndarray, 
#                         gridcell_size:float):
#     # pcd=o3d.geometry.PointCloud()    
#     # F2=np.where(np.isnan(pcloud_xyz).sum(1)==0)
#     # xyzF2=pcloud_xyz[F2]        
#     # pcd.points=o3d.utility.Vector3dVector(xyzF2)
#     pcloud_xyz=pcloud['xyz']
#     F2=F2=torch.isnan(pcloud_xyz).sum(1)==0
#     xyzF2=pcloud_xyz[F2]
#     weights=pcloud['probs'][F2]
#     minV=xyzF2[F2].min(0).values.cpu().numpy()

#     dbscan_eps=DBSCAN_EPS_MULTIPLIER*gridcell_size

#     minV=xyzF2[F2].min(0)
#     object_clusters=get_distinct_clusters(pcd, 
#                                     floor_threshold=minV[2],
#                                     cluster_min_count=ABSOLUTE_MIN_CLUSTER_SIZE,
#                                     gridcell_size=gridcell_size,
#                                     eps=dbscan_eps)  
    
#     # Merge clusters that are really close together
#     if MERGE_CLUSTER_MULTIPLIER>0:
#         list_count=10000
#         while len(object_clusters)<list_count:
#             m_clusters=merge_clusters(object_clusters, MERGE_CLUSTER_MULTIPLIER*gridcell_size)
#             object_clusters=m_clusters
#             list_count=len(object_clusters)
         
#     return object_clusters

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

def draw_boxes_around_cluster(fList_new:rgbd_file_list, 
                              fList_renders: rgbd_file_list, #(Optional - if not None, will draw boxes around both clusters in both images)
                                params:camera_params,
                                pcloud:dict,
                                cluster:object_pcloud
                                ):
    import cv2

    all_images={}
    rel_imgs=identify_related_images_global_pose(params,fList_new,cluster.centroid)
    for key in rel_imgs:
        iKey=int(key)
        fName=fList_new.get_color_fileName(iKey)
        all_images[iKey]={}

        # We only want images that had significant change identified
        #   so that we can build tight bounding boxes around the object of interest
        if fName not in pcloud['bboxes'] or len(pcloud['bboxes'][fName])==0:
            continue
        boxes=pcloud['bboxes'][fName]            
        M=fList_new.get_pose(iKey)
        sampled_points = np.array(cluster.find_pts_in_image(params,M,num_points=FARTHESTP_SAMPLE_SIZE))

        if len(sampled_points)<ABSOLUTE_MIN_CLUSTER_SIZE: # eliminate images that have too few points found in them
            continue
        
        box_count=[ count_points_in_box(sampled_points, box[1]) for box in boxes]

        prior=max(box_count)/sampled_points.shape[0]
        if prior<0.1:
            continue

        whichBox=np.argmax(box_count)        #box from sam that has the most number of points
        tgt_box=np.array(boxes[whichBox][1])

        # Expand bbox dimensions by 10-pt buffer
        new_box=np.hstack((truncate_point(tgt_box[:2]-10,params.width, params.height),
                            truncate_point(tgt_box[2:]+10,params.width, params.height)))
                
        colorI=cv2.imread(fName)
        colorI=cv2.rectangle(colorI, new_box[:2], new_box[2:], (255,0,0), 5)

        all_images[iKey]['new'] = colorI
        all_images[iKey]['max_prob'] = boxes[whichBox][0]
        all_images[iKey]['pt_count'] = box_count[whichBox]

        if fList_renders is not None:
            fName=fList_renders.get_color_fileName(iKey)
            colorI=cv2.imread(fName)
            colorI=cv2.rectangle(colorI, new_box[:2], new_box[2:], (255,0,0), 5)
            all_images[iKey]['render']=colorI
        
    return all_images

def build_change_clusters(pcloud_fileName,
                          scale_param=1.0 #optional scaling factor for the point cloud - set to 1.0 if not using synthetic depth data
                          ):
    import pickle
    pcloud = dict()
    object_clusters=[]

    try:
        with open(pcloud_fileName, 'rb') as handle:
            pcloud=pickle.load(handle)
    except Exception as e:
        print(f"pcloud file {pcloud_fileName} not found")
        return pcloud, []

    if pcloud['xyz'].shape[0]>ABSOLUTE_MIN_CLUSTER_SIZE:
        pcloud_xyz=pcloud['xyz']
        F2=F2=torch.isnan(pcloud_xyz).sum(1)==0
        xyzF2=pcloud_xyz[F2]
        weights=pcloud['probs'][F2]
        minV=xyzF2[F2].min(0).values.cpu().numpy()

        # pcloud_xyz=pcloud['xyz'].cpu().numpy()
        # Rescale everything ... 
        gridcell_size= 0.01/scale_param
        dbscan_eps=DBSCAN_EPS_MULTIPLIER*gridcell_size

        object_clusters=get_weighted_clusters(pcloud_xyz, 
                                              weights,
                                              floor_threshold=minV[2],
                                              cluster_min_count=10*ABSOLUTE_MIN_CLUSTER_SIZE,
                                              gridcell_size=gridcell_size,
                                              eps=dbscan_eps)
    return pcloud, object_clusters    

def write_change_cluster_images_to_disk(
        all_images, # from draw_boxes_around_cluster
        fList_new:rgbd_file_list,
        prompt,
        cluster_idx):
    import cv2
    file_prefix=prompt.replace(' ','_')
    for key in all_images:
        if 'new' in all_images[key]:
            fName_out=fList_new.intermediate_save_dir+f"/{file_prefix}_{cluster_idx}_{key}.png"
            print(fName_out)
            cv2.imwrite(fName_out,all_images[key]['new'] )
        

# def build_change_cluster_images(fList_new:rgbd_file_list, 
#                                 fList_renders:rgbd_file_list, 
#                                 params:camera_params,
#                                 pcloud_fileName, 
#                                 prompt,
#                                 scale_param=1.0 #optional scaling factor for the point cloud - set to 1.0 if not using synthetic depth data
#                                 ):
#     import pickle
#     import os
#     import cv2

#     try:
#         with open(pcloud_fileName, 'rb') as handle:
#             pcloud=pickle.load(handle)
#     except Exception as e:
#         print(f"pcloud file {pcloud_fileName} not found")
#         os._exit(-1)

#     file_prefix=prompt.replace(' ','_')
#     # Rescale everything ... 
#     if pcloud['xyz'].shape[0]>ABSOLUTE_MIN_CLUSTER_SIZE:
#         clusters=create_and_merge_clusters(pcloud['xyz'].cpu().numpy(), 0.01/scale_param)
#         clusters=merge_by_bounding_box(clusters, pcloud, fList_new, fList_renders, params)
#         for cluster_idx, cluster in enumerate(clusters):
#             all_images=draw_boxes_around_cluster(fList_new,None,params,pcloud,cluster)
#             for key in all_images:
#                 if fList_renders is not None:
#                     fName_out=fList_new.intermediate_save_dir+f"/{file_prefix}_{cluster_idx}_{key}.png"
#                 else:
#                     fName_out=fList_new.intermediate_save_dir+f"/{file_prefix}_{cluster_idx}_{key}.OV.png"
#                 print(fName_out)
#                 cv2.imwrite(fName_out,all_images[key]['main'] )
