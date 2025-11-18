# Utility functions for clustering point clouds.
# Includes:
#   - merge_clusters: merge nearby clusters based on distance
#   - merge_by_bounding_box: merge clusters based on bounding box overlap in images
#   - create_and_merge_clusters: create clusters from point cloud and merge nearby ones
#   - build_change_cluster_images: generate labeled images for each cluster of change
#   - count_points_in_box: count points within a bounding box
#   - truncate_point: truncate point coordinates to be within image bounds
#   - expand_bbox: expand a bounding box by a multiplier

from change_pcloud_utils.map_utils import identify_related_images_global_pose, get_distinct_clusters, object_pcloud
import numpy as np
import open3d as o3d
from change_pcloud_utils.rgbd_file_list import rgbd_file_list
from change_pcloud_utils.camera_params import camera_params

ABSOLUTE_MIN_CLUSTER_SIZE=100

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
     
def merge_by_bounding_box(clusters, 
                          pcloud,
                          fList_new:rgbd_file_list,
                          fList_renders:rgbd_file_list,
                          params:camera_params):
                          #exp_params):
    from shapely import Polygon
    cluster_image_dict=dict()

    for cluster_idx1, cluster in enumerate(clusters):
        rel_imgs=identify_related_images_global_pose(params,fList_new,cluster.centroid,None,0.5)
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
            sampled_points = np.array(cluster.find_pts_in_image(params,M,num_points=100))
            if len(sampled_points)<10:
                continue
            box_count=[ count_points_in_box(sampled_points, box[1]) for box in boxes]
            if max(box_count)>10:
                whichBox=np.argmax(box_count)
                tgt_box=np.array(boxes[whichBox][1])
                # Expand bbox dimensions by 1.5
                new_box=np.hstack((truncate_point(tgt_box[:2]-10,params.width, params.height),
                                    truncate_point(tgt_box[2:]+10,params.width, params.height))).astype(int)
                poly=Polygon(np.array([[new_box[0],new_box[1]],
                                       [new_box[0],new_box[3]],
                                       [new_box[2],new_box[3]],
                                       [new_box[2],new_box[1]],
                                       [new_box[0],new_box[1]]]))
                cluster_image_dict[cluster_idx1][iKey]=[fName,poly]         
    
    # Probababilistic scoring function
    scoring=dict()
    for cluster_idx1 in cluster_image_dict.keys():
        scoring[cluster_idx1]=dict()
        for cluster_idx2 in cluster_image_dict.keys():
            if cluster_idx1==cluster_idx2:
                continue
            scoring[cluster_idx1][cluster_idx2]=0

            for image_key1 in cluster_image_dict[cluster_idx1]:
                if image_key1 in cluster_image_dict[cluster_idx2]:
                    # Calculate IOU
                    if cluster_image_dict[cluster_idx1][image_key1][1].intersects(cluster_image_dict[cluster_idx2][image_key1][1]):
                        isect=cluster_image_dict[cluster_idx1][image_key1][1].intersection(cluster_image_dict[cluster_idx2][image_key1][1]).area
                        IOU=isect/(cluster_image_dict[cluster_idx1][image_key1][1].area + cluster_image_dict[cluster_idx2][image_key1][1].area - isect)
                    else:
                        IOU=0
                    IOU=min(0.95,max(0.05,IOU))
                else:
                    IOU=0.3
                # Update using log odds
                scoring[cluster_idx1][cluster_idx2]+=(np.log(IOU)-np.log(1-IOU))

    # Merge clusters based on scoring > 50% - need to make sure scoring agrees in both directions
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
            if  scoring[cluster_idx1][merge_candidate]>0 and scoring[merge_candidate][cluster_idx1]>0:
                isMerged[merge_candidate]=True
                exportCL=object_pcloud(np.vstack((exportCL.pts,clusters[merge_candidate].pts)),num_samples=100,sample=False)

        exportCL.sample_pcloud(100)
        merged_clusters.append(exportCL)
        isMerged[cluster_idx1]=True
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
        m_clusters=merge_clusters(object_clusters, 20*gridcell_size)
        object_clusters=m_clusters
        list_count=len(object_clusters)
         
    # pdb.set_trace()
    return object_clusters

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

def build_change_cluster_images(fList_new:rgbd_file_list, 
                                fList_renders:rgbd_file_list, 
                                params:camera_params,
                                pcloud_fileName, 
                                prompt,
                                scale_param=1.0 #optional scaling factor for the point cloud - set to 1.0 if not using synthetic depth data
                                ):
    import pickle
    import os
    import cv2

    try:
        with open(pcloud_fileName, 'rb') as handle:
            pcloud=pickle.load(handle)
    except Exception as e:
        print(f"pcloud file {pcloud_fileName} not found")
        os._exit(-1)

    file_prefix=prompt.replace(' ','_')
    # Rescale everything ... 
    if pcloud['xyz'].shape[0]>ABSOLUTE_MIN_CLUSTER_SIZE:
        clusters=create_and_merge_clusters(pcloud['xyz'].cpu().numpy(), 0.01/scale_param)
        clusters=merge_by_bounding_box(clusters, pcloud, fList_new, fList_renders, params)
        for cluster_idx, cluster in enumerate(clusters):
            rel_imgs=identify_related_images_global_pose(params,fList_new,cluster.centroid,None,0.5)
            for key in rel_imgs:
                iKey=int(key)
                fName=fList_new.get_color_fileName(iKey)

                # We only want images that had significant change identified
                #   so that we can build tight bounding boxes around the object of interest
                if fName not in pcloud['bboxes'] or len(pcloud['bboxes'][fName])==0:
                    continue

                boxes=pcloud['bboxes'][fName]            
                M=fList_new.get_pose(iKey)
                sampled_points = np.array(cluster.find_pts_in_image(params,M,num_points=100))
                if len(sampled_points)<10:
                    continue
                box_count=[ count_points_in_box(sampled_points, box[1]) for box in boxes]
                if max(box_count)>10:
                    whichBox=np.argmax(box_count)
                    tgt_box=np.array(boxes[whichBox][1])
                    # Expand bbox dimensions by 1.5
                    new_box=np.hstack((truncate_point(tgt_box[:2]-10,params.width, params.height),
                                       truncate_point(tgt_box[2:]+10,params.width, params.height)))
                    
                    colorI=cv2.imread(fName)
                    colorI=cv2.rectangle(colorI, new_box[:2], new_box[2:], (255,0,0), 5)

                    if fList_renders is not None:
                        fName_out=fList_new.intermediate_save_dir+f"/{file_prefix}_{cluster_idx}_{key}.png"
                    else:
                        fName_out=fList_new.intermediate_save_dir+f"/{file_prefix}_{cluster_idx}_{key}.OV.png"
                    print(fName_out)
                    cv2.imwrite(fName_out,colorI)
                    # cv2.imshow("image",colorI)
                    # cv2.waitKey(0)
