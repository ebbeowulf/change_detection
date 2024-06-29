import datetime
B1=datetime.datetime.now()
import torch
print(f"Library Load Time: {(datetime.datetime.now()-B1).total_seconds()}")
import pickle
import numpy as np
import cv2
import os
import pdb
from change_detection.segmentation import image_segmentation
from rgbd_file_list import rgbd_file_list
from camera_params import camera_params
import copy

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# Process all images with yolo - creating
#   pickle files for the results and storing alongside the 
#   original color images
def process_images_with_yolo(fList:rgbd_file_list):
    print("process_images")
    from change_detection.yolo_segmentation import yolo_segmentation
    YS=yolo_segmentation()
    for key in fList.keys():
        print(fList.get_color_fileName(key))
        pkl_fName=fList.get_yolo_fileName(key)
        if not os.path.exists(pkl_fName):
            img=YS.process_file(fList.get_color_fileName(key),save_fileName=pkl_fName)

def process_images_with_clip(fList:rgbd_file_list, clip_targets:list):
    print("process_images")
    from change_detection.clip_segmentation import clip_seg
    YS=clip_seg(clip_targets)
    for key in fList.keys():
        print(fList.get_color_fileName(key))
        for target in clip_targets:
            pkl_fName=fList.get_clip_fileName(key,target)
            if not os.path.exists(pkl_fName):
                img=YS.process_file(fList.get_color_fileName(key),save_fileName=pkl_fName)

# Create a list of all of the objects recognized by yolo
#   across all files. Will only load existing pkl files, not 
#   create any new ones
def clip_threshold_evaluation(fList:rgbd_file_list, clip_targets:list, proposed_threshold:float):
    from change_detection.clip_segmentation import clip_seg

    YS=clip_seg(clip_targets)
    image_list=[]
    for target in clip_targets:
        maxP=[]
        for key in fList.keys():
            # Use a high threshold here so that we are not creating DBScan boxes unnecessarily
            if not YS.load_file(fList.get_clip_fileName(key,target),threshold=proposed_threshold):
                continue
            P=YS.get_max_prob(target)
            if P is not None:
                maxP.append(P)
                if P>proposed_threshold:
                    image_list.append(key)
        count=(np.array(maxP)>proposed_threshold).sum()
        print("%s: Counted %d / %d images with detections > %f"%(target, count,len(maxP),proposed_threshold))
    return np.unique(image_list).tolist()
# Create a list of all of the objects recognized by yolo
#   across all files. Will only load existing pkl files, not 
#   create any new ones
def create_yolo_object_list(fList:rgbd_file_list):
    from change_detection.yolo_segmentation import yolo_segmentation

    YS=yolo_segmentation()
    obj_list=dict()
    for key in fList.keys():
        if not YS.load_file(fList.get_yolo_fileName(key)):
            continue
        
        for id in YS.boxes.keys():
            if YS.id2label[id] not in obj_list:
                obj_list[YS.id2label[id]]={'images': [], 'probs': []}
            for box in YS.boxes[id]:
                obj_list[YS.id2label[id]]['images'].append(key)
                obj_list[YS.id2label[id]]['probs'].append(box[0])
    return obj_list

# Reprocess the object list (above) to return only the set of 
#   objects that exceed the provided threshold
def get_high_confidence_objects(obj_list, confidence_threshold=0.5):
    o_list=[]
    for key in obj_list:
        maxV=max(obj_list[key]['probs'])
        if maxV>=confidence_threshold:
            o_list.append(key)
    return o_list

# Create an open3d pointcloud object - will randomly sample the cloud
#   to reduce the number of points as necessary
def pointcloud_open3d(xyz_points,rgb_points=None,max_num_points=2000000):
    import open3d as o3d
    pcd=o3d.geometry.PointCloud()
    if xyz_points.shape[0]<max_num_points:
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        if rgb_points is not None:
            pcd.colors = o3d.utility.Vector3dVector(rgb_points[:,[2,1,0]]/255) 
    else:
        rr=np.random.choice(np.arange(xyz_points.shape[0]),max_num_points)
        pcd.points = o3d.utility.Vector3dVector(xyz_points[rr,:])
        if rgb_points is not None:
            rgb2=rgb_points[rr,:]
            pcd.colors = o3d.utility.Vector3dVector(rgb2[:,[2,1,0]]/255) 

    return pcd

# Combine together multiple point clouds into a single
#   cloud and display the result using open3d
def visualize_combined_xyzrgb(fList:rgbd_file_list, params:camera_params, howmany_files=100, skip=0, max_num_points=2000000):
    rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy)
    cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx)

    combined_xyz=np.zeros((0,3),dtype=float)
    combined_rgb=np.zeros((0,3),dtype=np.uint8)

    rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)

    howmany=0
    count=0
    for key in range(max(fList.keys())):
        if not fList.is_key(key):
            continue
        count+=1
        if count<=skip:
            continue
        count=0
        # Create the generic depth data
        print(fList.get_color_fileName(key))
        colorI=cv2.imread(fList.get_color_fileName(key), -1)
        depthI=cv2.imread(fList.get_depth_fileName(key), -1)
        depthT=torch.tensor(depthI.astype('float')/1000.0)
        colorT=torch.tensor(colorI)
        x = cols*depthT/params.fx
        y = rows*depthT/params.fy
        depth_mask=(depthT>1e-4)*(depthT<10.0)

        # Rotate the points into the right space
        M=torch.matmul(rot_matrixT,torch.tensor(fList.get_pose(key),device=DEVICE))
        pts=torch.stack([x[depth_mask],y[depth_mask],depthT[depth_mask],torch.ones(((depth_mask>0).sum()))],dim=1)
        pts_rot=torch.matmul(M,pts.transpose(0,1))
        pts_rot=pts_rot[:3,:].transpose(0,1)

        if pts_rot.shape[0]>100:
            combined_xyz=np.vstack((combined_xyz,pts_rot.cpu().numpy()))
            combined_rgb=np.vstack((combined_rgb,colorT[depth_mask].cpu().numpy()))
        howmany+=1
        if howmany==howmany_files:
            break
    return pointcloud_open3d(combined_xyz,rgb_points=combined_rgb, max_num_points=max_num_points)

# Agglomerative clustering - not optimized
#   combines points together based on distance
#   rather slow on large point clouds
def agglomerative_cluster(pts, max_dist):
    clusters=[]
    for idx in range(pts.shape[0]):
        clusters.append({'pts': pts[idx,:].reshape((1,3)), 'found': False})
    
    num_old=0
    while num_old!=len(clusters):
        print("Clustering - %d clusters"%(len(clusters)))
        new_clusters=[]
        num_old=len(clusters)
        for idx in range(len(clusters)):
            # Skip clusters that have already been processed
            if clusters[idx]['found']:
                continue

            for jdx in range(idx+1,len(clusters)):
                # Skip clusters that have already been processed
                if clusters[jdx]['found']:
                    continue
                pairwise_distance=torch.cdist(clusters[idx]['pts'],clusters[jdx]['pts'])
                if pairwise_distance.min()<max_dist:
                    # Merge clusters
                    clusters[idx]['pts']=torch.vstack((clusters[idx]['pts'],clusters[jdx]['pts']))
                    clusters[jdx]['found']=True
            new_clusters.append(clusters[idx])
        clusters=new_clusters   
    return clusters     

# A filter based on connected components - it starts from a centroid and grows outward with random sampling.
#   Tested with and without gpu - faster with cpu only, but still longer than desired
def connected_components_filter(centerRC, depthT:torch.tensor, maskI:torch.tensor, neighborhood=4, max_depth_dist=0.1):
    queue=[centerRC]
    cc_mask=torch.zeros(maskI.shape,dtype=torch.uint8,device=DEVICE)
    cc_mask[centerRC[0],centerRC[1]]=2
    iterations=0
    while len(queue)>0:
        point=queue.pop(0)
        target_depth = depthT[point[0],point[1]]
        
        minR=max(0,point[0]-neighborhood)
        minC=max(0,point[1]-neighborhood)
        maxR=min(depthT.shape[0],point[0]+neighborhood)
        maxC=min(depthT.shape[1],point[1]+neighborhood)
        regionD=depthT[minR:maxR,minC:maxC]
        regionMask=maskI[minR:maxR,minC:maxC]
        regionCC=cc_mask[minR:maxR,minC:maxC]

        reachableAreaMask=((regionD-target_depth).abs()<max_depth_dist)*regionMask
        localMask=reachableAreaMask*(regionCC==0)

        # Update the queue
        if 1: # randomsample
            sample_size=5
            indices=localMask.nonzero()+torch.tensor([minR,minC],device=DEVICE)
            if len(indices)<sample_size:
                queue=queue+indices.tolist()
            else:
                rr=np.random.choice(np.arange(len(indices)),sample_size)
                queue=queue+indices[rr].tolist()
        else:
            indices=localMask.nonzero()+torch.tensor([minR,minC],device=DEVICE)            
            queue=queue+indices.cpu().tolist()

        # Set all points in the cc_mask so that we don't keep looking at them
        regionCC[reachableAreaMask]=2
        regionCC[reachableAreaMask==0]=1
        iterations+=1
        if iterations % 500==0:
            print("Iterations=%d"%(iterations))
    print("Iterations=%d"%(iterations))
    return cc_mask==2

# Find a valid center point of a bounding box. The neighborhood indicates
#   the size of the area to search around the mathematical centroid.
def get_center_point(depthT:torch.tensor, combo_mask:torch.tensor, xy_bbox, neighborhood=20):
    rowC=int((xy_bbox[3]-xy_bbox[1])/2.0 + xy_bbox[1])
    colC=int((xy_bbox[2]-xy_bbox[0])/2.0 + xy_bbox[0])
    minR=max(0,rowC-5)
    minC=max(0,colC-5)
    maxR=min(depthT.shape[0],rowC+neighborhood)
    maxC=min(depthT.shape[1],colC+neighborhood)
    regionDepth=depthT[minR:maxR,minC:maxC]
    regionMask=combo_mask[minR:maxR,minC:maxC]
    mean_depth=regionDepth[regionMask].mean()
    indices = regionMask.nonzero()
    dist=(indices-torch.tensor([neighborhood,neighborhood],device=DEVICE)).pow(2).sum(1).sqrt()*0.1 + (regionDepth[regionMask]-mean_depth).abs()
    if dist.shape[0]==0:
        return None
    whichD=dist.argmin()
    return (indices[whichD].cpu()+torch.tensor([minR,minC])).tolist()

def get_rotated_points(x, y, depth, filtered_maskT, rot_matrix):
    pts=torch.stack([x[filtered_maskT],
                        y[filtered_maskT],
                        depth[filtered_maskT],
                        torch.ones(((filtered_maskT>0).sum()),
                        device=DEVICE)],dim=1)
    pts_rot=torch.matmul(rot_matrix,pts.transpose(0,1))
    return pts_rot[:3,:].transpose(0,1)

# Create pclouds for all of the indicated target classes, saving the resulting cloud to the
#   disk so that it does not need to be re-calculated each time. Note that the connected components
#   filter is being applied each time.
def create_pclouds(tgt_classes:list, fList:rgbd_file_list, params:camera_params, is_yolo:bool, conf_threshold=0.5, use_connected_components=True):
    if is_yolo:
        from change_detection.yolo_segmentation import yolo_segmentation
        YS=yolo_segmentation()
    else:
        from change_detection.clip_segmentation import clip_seg
        YS=clip_seg(tgt_classes)

    rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
    cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)

    pclouds=dict()
    for cls in tgt_classes:
        pclouds[cls]={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8),'probs': []}

    image_key_list=clip_threshold_evaluation(fList, tgt_classes, conf_threshold)

    rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)
    # for key in range(max(fList.keys())):
    #     if not fList.is_key(key):
    #         continue
    from datetime import datetime, timedelta
    all_times=[]
    for key in image_key_list:
        print(key)
        try:
            # T0=datetime.now()
            colorI=cv2.imread(fList.get_color_fileName(key), -1)
            depthI=cv2.imread(fList.get_depth_fileName(key), -1)
            depthT=torch.tensor(depthI.astype('float')/1000.0,device=DEVICE)
            colorT=torch.tensor(colorI,device=DEVICE)
            x = cols*depthT/params.fx
            y = rows*depthT/params.fy
            depth_mask=(depthT>1e-4)*(depthT<10.0)

            # Build the rotation matrix
            M=torch.matmul(rot_matrixT,torch.tensor(fList.get_pose(key),device=DEVICE))

            # T1=datetime.now()

            # Now extract a mask per category
            for cls in tgt_classes:
                # T2=datetime.now()
                # Try to load the file
                threshold=conf_threshold*2/3
                if not YS.load_file(fList.get_segmentation_fileName(key, is_yolo, cls),threshold=threshold):
                    continue
                cls_mask=YS.get_mask(cls)
                # T3=datetime.now()
                if cls_mask is not None and YS.get_max_prob(cls)>=conf_threshold:
                    if use_connected_components:
                        save_fName=fList.get_class_pcloud_fileName(key,cls)
                        # Load or create the connected components mask for this object type
                        try:
                            if os.path.exists(save_fName):
                                with open(save_fName, 'rb') as handle:
                                    filtered_maskT=pickle.load(handle)
                            else:
                                # We need to build the boxes around clusters with clip-based segmentation
                                #   YOLO should already have the boxes in place
                                if YS.get_boxes(cls) is None or len(YS.get_boxes(cls))==0:
                                    YS.build_dbscan_boxes(cls,threshold=threshold)
                                # If this is still zero ...
                                if len(YS.get_boxes(cls))<1:
                                    continue
                                combo_mask=(torch.tensor(cls_mask,device=DEVICE)>conf_threshold)*depth_mask
                                # Find the list of boxes associated with this object
                                boxes=YS.get_boxes(cls)
                                filtered_maskT=None
                                for box in boxes:
                                    # Pick a point from the center of the mask to use as a centroid...
                                    ctrRC=get_center_point(depthT, combo_mask, box[1])
                                    if ctrRC is None:
                                        continue

                                    maskT=connected_components_filter(ctrRC,depthT, combo_mask, neighborhood=10)
                                    # Combine masks from multiple objects
                                    if filtered_maskT is None:
                                        filtered_maskT=maskT
                                    else:
                                        filtered_maskT=filtered_maskT*maskT

                                with open(save_fName,'wb') as handle:
                                    pickle.dump(filtered_maskT, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        except Exception as e:
                            pdb.set_trace()
                            print("Exception " + str(e) +" - skipping")
                            continue
                    else:
                        filtered_maskT=(torch.tensor(cls_mask,device=DEVICE)>conf_threshold)*depth_mask
                    # T4=datetime.now()

                    pts_rot=get_rotated_points(x,y,depthT,filtered_maskT,M) 
                    if pts_rot.shape[0]>100:
                        pclouds[cls]['xyz']=np.vstack((pclouds[cls]['xyz'],pts_rot.cpu().numpy()))
                        pclouds[cls]['rgb']=np.vstack((pclouds[cls]['rgb'],colorT[filtered_maskT].cpu().numpy()))
                        pclouds[cls]['probs']=np.hstack((pclouds[cls]['probs'],YS.get_prob_array(cls)[filtered_maskT]))
                    # T5=datetime.now()

        except Exception as e:
            continue
        # TIME_ARRAY=[(T1-T0).total_seconds(),(T3-T2).total_seconds(),(T4-T3).total_seconds(),(T5-T4).total_seconds()]
        # all_times.append(TIME_ARRAY)
        # if len(all_times)>50:
        #     pdb.set_trace()

    return pclouds

def create_pclouds_from_images(fList:rgbd_file_list, params:camera_params, targets:list=None, display_pclouds=False, use_connected_components=False):
    import open3d as o3d
    process_images_with_yolo(fList)
    if targets is None:
        obj_list=create_object_list(fList)
        print("Detected Objects (Conf > 0.75)")
        high_conf_list=get_high_confidence_objects(obj_list, confidence_threshold=0.75)
        print(high_conf_list)
        pclouds=create_pclouds(high_conf_list,fList,params,conf_threshold=0.5,use_connected_components=use_connected_components)
    else:
        pclouds=create_pclouds(targets,fList,params,conf_threshold=0.5,use_connected_components=use_connected_components)
    for key in pclouds.keys():
        fileName=fList.intermediate_save_dir+"/"+key+".ply"
        pcd=pointcloud_open3d(pclouds[key]['xyz'],pclouds[key]['rgb'])
        o3d.io.write_point_cloud(fileName,pcd)
        if display_pclouds:
            o3d.visualization.draw_geometries([pcd])
