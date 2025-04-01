import datetime
B1=datetime.datetime.now()
import torch
print(f"Library Load Time: {(datetime.datetime.now()-B1).total_seconds()}")
import pickle
import numpy as np
import cv2
import os
import pdb
import sys
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'change_detection', 'scripts'))
sys.path.append(scripts_path)
from change_detection.segmentation import image_segmentation
from rgbd_file_list import rgbd_file_list
from camera_params import camera_params
import copy

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# Process all images with omdet - creating
#   pickle files for the results and storing alongside the 
#   original color images

def process_images_with_omdet(fList:rgbd_file_list, clip_targets:list):
    print("process_images")
    from change_detection.omdet_segmentation import omdet_seg
    YS=omdet_seg(clip_targets)
    for key in fList.keys():
        print(fList.get_color_fileName(key))
        for target in clip_targets:
            pkl_fName=fList.get_omdet_fileName(key,target)
            if not os.path.exists(pkl_fName):
                img=YS.process_file(fList.get_color_fileName(key),save_fileName=pkl_fName)

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

def omdet_threshold_evaluation(fList:rgbd_file_list, clip_targets:list, proposed_threshold:float):
    from change_detection.omdet_segmentation import omdet_seg
    YS=omdet_seg(clip_targets)
    image_list=[]
    for target in clip_targets:
        maxP=[]
        for key in fList.keys():
            # Use a high threshold here so that we are not creating DBScan boxes unnecessarily
            # load_file now returns True/False for success/failure
            if not YS.load_file(fList.get_omdet_fileName(key,target)):
                continue
            # We need a way to get relevant scores or decide if the frame is useful without a threshold here.
            # Placeholder: Assume if loading succeeded, the frame *might* be useful.
            # A better approach might involve checking if YS.scores[target_index] is non-empty after loading.
            # For now, let's just add the key if load succeeds.
            # TODO: Revisit this logic if needed.
            if YS.scores: # Check if any scores were loaded
                 target_index = YS.label2id.get(target)
                 if target_index is not None and YS.scores.get(target_index): # Check scores for specific target
                    # Simple check: add key if scores exist for this target in this frame
                    # More complex check could involve max(YS.scores[target_index]) > some_threshold
                    if key not in image_list:
                        image_list.append(key)

        # This part calculating maxP seems related to the old threshold logic and might not be needed
        # or needs adjustment based on stored scores.
        # Commenting out for now as it relies on 'P' which wasn't calculated correctly.
        # if maxP:
        #     maxP=np.array(maxP)
        #     pThresh=np.percentile(maxP,90)
        #     if proposed_threshold<pThresh:
        #         print("Proposed Threshold: "+str(proposed_threshold)+" -> Percentile Thresh: "+str(pThresh))
        #         proposed_threshold=pThresh
        #         image_list=[im for p,im in zip(maxP,image_list) if p>proposed_threshold]

    return image_list

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
        count += 1
        if count <= skip:
            continue
        count = 0
        # Load images
        print(fList.get_color_fileName(key))
        colorI = cv2.imread(fList.get_color_fileName(key), -1)
        depthI = cv2.imread(fList.get_depth_fileName(key), -1)
        
        # Check if images loaded successfully (optional but recommended)
        if colorI is None or depthI is None:
            print(f"Failed to load images for key {key}")
            continue
        
        # Resize depth image to match the resolution defined in params
        depthI_resized = cv2.resize(depthI, (params.width, params.height), interpolation=cv2.INTER_NEAREST)
        
        # Convert to tensor and scale depth (mm to meters)
        depthT = torch.tensor(depthI_resized.astype('float') / 1000.0)
        colorT = torch.tensor(colorI)
        
        # Compute 3D coordinates
        x = cols * depthT / params.fx
        y = rows * depthT / params.fy
        depth_mask = (depthT > 1e-4) * (depthT < 10.0)

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

class pcloud_from_images():
    def __init__(self,fList:rgbd_file_list, params:camera_params, default_pcloud_threshold=0.5):
        self.fList=fList
        self.params=params
        self.default_threshold=default_pcloud_threshold
        self.YS=None
        self.rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
        self.cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)
        self.rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)        
        self.loaded_image=None
        print(f"pcloud_from_images initialized. Default Threshold: {self.default_threshold}") # Added print

    # Image loading to allow us to process more than one class in rapid succession
    def load_image(self, image_key, max_distance=10.0):
        if self.loaded_image is None or self.loaded_image['key']!=image_key:
            try:
                if self.loaded_image is None:
                    self.loaded_image=dict()
                colorI=cv2.imread(self.fList.get_color_fileName(image_key), -1)
                depthI=cv2.imread(self.fList.get_depth_fileName(image_key), -1)
                self.loaded_image['depthT']=torch.tensor(depthI.astype('float')/1000.0,device=DEVICE)
                self.loaded_image['colorT']=torch.tensor(colorI,device=DEVICE)
                self.loaded_image['x'] = self.cols*self.loaded_image['depthT']/self.params.fx
                self.loaded_image['y'] = self.rows*self.loaded_image['depthT']/self.params.fy
                self.loaded_image['depth_mask']=(self.loaded_image['depthT']>1e-4)*(self.loaded_image['depthT']<max_distance)

                # Build the rotation matrix
                self.loaded_image['M']=torch.matmul(self.rot_matrixT,torch.tensor(self.fList.get_pose(image_key),device=DEVICE))

                # Save the key last so we can skip if called again
                self.loaded_image['key']=image_key

                print(f"Image loaded: {image_key}")
                return True
            except Exception as e:
                print(f"Failed to load image materials for {image_key}")
                self.loaded_image=None
            return False
        return True
    
    def process_image(self, image_key, tgt_class, conf_threshold, use_connected_components=False):
        # Create the image segmentation file
        if self.YS is None or tgt_class not in self.YS.get_all_classes():
            from change_detection.clip_segmentation import clip_seg
            self.YS=clip_seg([tgt_class])

        # Load the necessary RGBD and segmentation files
        if not self.load_image(image_key):
            return None

        # Recover the segmentation file
        if not self.YS.load_file(self.fList.get_segmentation_fileName(image_key, False, tgt_class),threshold=conf_threshold):
            return None

        # Build the class associated mask for this image
        cls_mask=self.YS.get_mask(tgt_class)

        # Apply connected components if requested       
        if use_connected_components:
            filtered_maskT=self.cluster_pcloud()
        else:
            filtered_maskT=(torch.tensor(cls_mask,device=DEVICE)>conf_threshold)*self.loaded_image['depth_mask']

        # Return all points associated with the target class
        pts_rot=get_rotated_points(self.loaded_image['x'],self.loaded_image['y'],self.loaded_image['depthT'],filtered_maskT,self.loaded_image['M']) 
        return {'xyz': pts_rot.cpu().numpy(), 
                'rgb': self.loaded_image['colorT'][filtered_maskT].cpu().numpy(), 
                'probs': self.YS.get_prob_array(tgt_class)[filtered_maskT]}
        
    #Apply clustering - slow... probably in need of repair
    def cluster_pclouds(self, image_key, tgt_class, cls_mask, threshold):
        save_fName=self.fList.get_class_pcloud_fileName(image_key,tgt_class)
        if os.path.exists(save_fName):
            with open(save_fName, 'rb') as handle:
                filtered_maskT=pickle.load(handle)
        else:
            # We need to build the boxes around clusters with clip-based segmentation
            #   YOLO should already have the boxes in place
            if self.YS.get_boxes(tgt_class) is None or len(self.YS.get_boxes(tgt_class))==0:
                self.YS.build_dbscan_boxes(tgt_class,threshold=threshold)
            # If this is still zero ...
            if len(self.YS.get_boxes(tgt_class))<1:
                return None
            combo_mask=(torch.tensor(cls_mask,device=DEVICE)>threshold)*self.loaded_image['depth_mask']
            # Find the list of boxes associated with this object
            boxes=self.YS.get_boxes(tgt_class)
            filtered_maskT=None
            for box in boxes:
                # Pick a point from the center of the mask to use as a centroid...
                ctrRC=get_center_point(self.loaded_image['depthT'], combo_mask, box[1])
                if ctrRC is None:
                    continue

                maskT=connected_components_filter(ctrRC,self.loaded_image['depthT'], combo_mask, neighborhood=10)
                # Combine masks from multiple objects
                if filtered_maskT is None:
                    filtered_maskT=maskT
                else:
                    filtered_maskT=filtered_maskT*maskT
            with open(save_fName,'wb') as handle:
                pickle.dump(filtered_maskT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return filtered_maskT

    # Process a sequence of images, combining the resulting point clouds together
    #   The combined result will be saved to disk for faster retrieval
    def process_fList(self, tgt_class, conf_threshold, use_connected_components=False):
        save_fName=self.fList.get_combined_raw_fileName(tgt_class)
        pcloud=None
        if os.path.exists(save_fName):
            try:
                with open(save_fName, 'rb') as handle:
                    pcloud=pickle.load(handle)
            except Exception as e:
                pcloud=None
                print("Failed to load save file - rebuilding... " + save_fName)
        
        if pcloud is None:
            # Build the pcloud from individual images
            pcloud={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8),'probs': np.array([], dtype=float)}
            
            image_key_list=omdet_threshold_evaluation(self.fList, [tgt_class], self.default_threshold)
            for key in image_key_list:
                icloud=self.process_image(key, tgt_class, self.default_threshold, use_connected_components=use_connected_components)
                if icloud is not None and icloud['xyz'].shape[0]>100:
                    pcloud['xyz']=np.vstack((pcloud['xyz'],icloud['xyz']))
                    pcloud['rgb']=np.vstack((pcloud['rgb'],icloud['rgb']))
                    pcloud['probs']=np.hstack((pcloud['probs'],icloud['probs']))
            
            # Now save the result so we don't have to keep processing this same cloud
            with open(save_fName,'wb') as handle:
                pickle.dump(pcloud, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Finally - filter the cloud with the requested confidence threshold
        if pcloud['probs']:
            whichP=(pcloud['probs']>conf_threshold)
            return {'xyz':pcloud['xyz'][whichP],'rgb':pcloud['rgb'][whichP],'probs':pcloud['probs'][whichP]}
        else:
            return {'xyz':pcloud['xyz'],'rgb':pcloud['rgb'],'probs':pcloud['probs']}

def create_pclouds_from_images(fList:rgbd_file_list, params:camera_params, targets:list=None, display_pclouds=False, use_connected_components=False):
    import open3d as o3d
    process_images_with_clip(fList)
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

def build_scene_pcd(data_path, labels, output_dir,raw_dir="raw_output", save_dir="save_results", depth_scale=1000.0, draw=False):
    print(f"Building scene PCD from: {data_path}")
    print(f"Labels: {labels}")
    # Create pcloud_from_images instances for each label + one for full cloud
    pcd_builder = pcloud_from_images(fList, labels=None, depth_scale=depth_scale) # For full cloud
    pcd_builders_labeled = {label: pcloud_from_images(fList, labels=[label], targetClass=label, depth_scale=depth_scale) for label in labels}

    # Process images and build point clouds
    print("\nBuilding full point cloud...")
    full_pcd = pcd_builder.process_fList()
    if full_pcd and len(full_pcd.points)>0:
         o3d.io.write_point_cloud(os.path.join(save_dir_path, f"combined.ply"), full_pcd)
         print(f"Saved full point cloud to combined.ply")
    else:
         print("Full point cloud generation resulted in 0 points or None.")

    labeled_pcds = {}
    for label in labels:
        print(f"\nBuilding point cloud for label: '{label}'...")
        builder = pcd_builders_labeled[label]
        labeled_pcd = builder.process_fList()

        if labeled_pcd and len(labeled_pcd.points)>0 :
            print(f"Successfully generated {len(labeled_pcd.points)} points for '{label}'.")
            labeled_pcds[label] = labeled_pcd
            # Save individual labeled PCD
            save_path = os.path.join(save_dir_path, f"{label.replace(' ', '_')}.labeled.ply") # Replace spaces for filename
            o3d.io.write_point_cloud(save_path, labeled_pcd)
            print(f"Saved labeled point cloud for '{label}' to {save_path}")
        else:
            print(f"Failed to generate point cloud for label '{label}' (0 points or None returned).") # Modified print

    # Visualization (Optional)
    if draw:
        o3d.visualization.draw_geometries([full_pcd] + list(labeled_pcds.values()))
