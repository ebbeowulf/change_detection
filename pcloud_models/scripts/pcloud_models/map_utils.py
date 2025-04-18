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
from sklearn.cluster import DBSCAN
from farthest_point_sampling.fps import farthest_point_sampling
import time

DBSCAN_MIN_SAMPLES=5 
DBSCAN_GRIDCELL_SIZE=0.05
DBSCAN_EPS=0.05 # allows for connections in cells full range of surrounding cube DBSCAN_GRIDCELL_SIZE*2.5
CLUSTER_MIN_COUNT=5000
CLUSTER_PROXIMITY_THRESH=0.3
CLUSTER_TOUCHING_THRESH=0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# Process all images with yolo - creating
#   pickle files for the results and storing alongside the 
#   original color images
# def process_images_with_yolo_world(fList:rgbd_file_list, targets:list):
#     print("process_images")
#     from change_detection.yolo_world_segmentation import yolo_segmentation
#     YS=yolo_segmentation(targets ,'yolov8x-worldv2.pt')
#     for key in fList.keys():
#         for target in targets:
#             pkl_fName=fList.get_yolo_fileName(key,target)
#             if not os.path.exists(pkl_fName):
#                 img=YS.process_file(fList.get_color_fileName(key),save_fileName=pkl_fName)

# def process_images_with_clip(fList:rgbd_file_list, clip_targets:list):
#     print("process_images")
#     from change_detection.clip_segmentation import clip_seg
#     YS=clip_seg(clip_targets)
#     for key in fList.keys():
#         print(fList.get_color_fileName(key))
#         for target in clip_targets:
#             pkl_fName=fList.get_clip_fileName(key,target)
#             if not os.path.exists(pkl_fName):
#                 img=YS.process_file(fList.get_color_fileName(key),save_fileName=pkl_fName)

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

# def yolo_world_threshold_evaluation(fList:rgbd_file_list, yolo_targets:list, proposed_threshold:float):
#     from change_detection.yolo_world_segmentation import yolo_segmentation
    
#     YS=yolo_segmentation(yolo_targets)
#     image_list=[]
#     #pdb.set_trace()
#     for target in yolo_targets:
#         maxP=[]
#         for key in fList.keys():
#             image_list.append(key)
#     return np.unique(image_list).tolist()

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
    filtered_maskT = filtered_maskT.bool()
    pts=torch.stack([x[filtered_maskT],
                        y[filtered_maskT],
                        depth[filtered_maskT],
                        torch.ones(((filtered_maskT>0).sum()),
                        device=DEVICE)],dim=1)
    pts_rot=torch.matmul(rot_matrix,pts.transpose(0,1))
    return pts_rot[:3,:].transpose(0,1)

class pcloud_from_images():
    def __init__(self, params:camera_params):
        self.params=params
        self.YS=None
        self.rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
        self.cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)
        self.rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)        
        self.loaded_image=None
        self.classifier_type=None

    # Image loading to allow us to process more than one class in rapid succession
    def load_image_from_file(self, fList:rgbd_file_list, image_key, max_distance=10.0):
        colorI=cv2.imread(fList.get_color_fileName(image_key), -1)
        depthI=cv2.imread(fList.get_depth_fileName(image_key), -1)
        poseM=fList.get_pose(image_key)
        self.load_image(colorI, depthI, poseM, image_key, max_distance=max_distance)

    def load_image(self, colorI:np.ndarray, depthI:np.ndarray, poseM:np.ndarray, uid_key:str, max_distance=10.0):
        if self.loaded_image is None or self.loaded_image['key']!=uid_key:
            try:
                if self.loaded_image is None:
                    self.loaded_image=dict()
                self.loaded_image['depthT']=torch.tensor(depthI.astype('float')/1000.0,device=DEVICE)
                self.loaded_image['colorT']=torch.tensor(colorI,device=DEVICE)
                self.loaded_image['x'] = self.cols*self.loaded_image['depthT']/self.params.fx
                self.loaded_image['y'] = self.rows*self.loaded_image['depthT']/self.params.fy
                self.loaded_image['depth_mask']=(self.loaded_image['depthT']>1e-4)*(self.loaded_image['depthT']<max_distance)

                # Build the rotation matrix
                self.loaded_image['M']=torch.matmul(self.rot_matrixT,torch.tensor(poseM,device=DEVICE))

                # Save the key last so we can skip if called again
                self.loaded_image['key']=uid_key

                print(f"Image loaded: {uid_key}")
                return True
            except Exception as e:
                print(f"Failed to load image materials for {uid_key}")
                self.loaded_image=None
            return False
        return True

    def get_pts_per_class(self, tgt_class, use_connected_components=False, rotate90=False):
        # Build the class associated mask for this image
        cls_mask=self.YS.get_mask(tgt_class)
        if cls_mask is not None:
            if type(cls_mask)==torch.Tensor:
                if rotate90:
                    cls_maskT=torch.rot90(cls_mask,dims=(0,1))
                else:
                    cls_maskT=cls_mask
            else:
                if rotate90:
                    cls_maskT=torch.tensor(np.rot90(cls_mask,dims=(0,1)).copy(),device=DEVICE)
                else:
                    cls_maskT=torch.tensor(cls_mask,device=DEVICE)

            # Apply connected components if requested       
            if use_connected_components:
                filtered_maskT=self.cluster_pcloud()
            else:
                filtered_maskT=cls_maskT*self.loaded_image['depth_mask']

            # Return all points associated with the target class
            pts_rot=get_rotated_points(self.loaded_image['x'],self.loaded_image['y'],self.loaded_image['depthT'],filtered_maskT,self.loaded_image['M']) 
            filtered_maskT = filtered_maskT.bool()
            if rotate90:
                probs=torch.rot90(self.YS.get_prob_array(tgt_class),dims=(0,1))
                return {'xyz': pts_rot, 
                        'rgb': self.loaded_image['colorT'][filtered_maskT], 
                        'probs': probs[filtered_maskT]}
            else:
                return {'xyz': pts_rot, 
                        'rgb': self.loaded_image['colorT'][filtered_maskT], 
                        'probs': self.YS.get_prob_array(tgt_class)[filtered_maskT]}
        
        else:
            return None

    def get_pts_per_class2(self, tgt_class, use_connected_components=False, rotate90=False):
        # Build the class associated mask for this image
        cls_mask=self.YS2.get_mask(tgt_class)
        if cls_mask is not None:
            if type(cls_mask)==torch.Tensor:
                if rotate90:
                    cls_maskT=torch.rot90(cls_mask,dims=(0,1))
                else:
                    cls_maskT=cls_mask
            else:
                if rotate90:
                    cls_maskT=torch.tensor(np.rot90(cls_mask,dims=(0,1)).copy(),device=DEVICE)
                else:
                    cls_maskT=torch.tensor(cls_mask,device=DEVICE)

            # Apply connected components if requested       
            if use_connected_components:
                filtered_maskT=self.cluster_pcloud()
            else:
                filtered_maskT=cls_maskT*self.loaded_image['depth_mask']

            # Return all points associated with the target class
            pts_rot=get_rotated_points(self.loaded_image['x'],self.loaded_image['y'],self.loaded_image['depthT'],filtered_maskT,self.loaded_image['M']) 
            filtered_maskT = filtered_maskT.bool()
            if rotate90:
                probs=torch.rot90(self.YS2.get_prob_array(tgt_class),dims=(0,1))
                return {'xyz': pts_rot, 
                        'rgb': self.loaded_image['colorT'][filtered_maskT], 
                        'probs': probs[filtered_maskT]}
            else:
                return {'xyz': pts_rot, 
                        'rgb': self.loaded_image['colorT'][filtered_maskT], 
                        'probs': self.YS2.get_prob_array(tgt_class)[filtered_maskT]}
        
    def setup_image_processing(self, tgt_class_list, classifier_type):
        # Check to see if the classifier already exists AND if it has 
        #   all of the necessary files in its class list
        is_update_required=False
        if self.YS is None or classifier_type!=self.classifier_type:
            is_update_required=True
        else:
            for tgt in tgt_class_list:
                if tgt not in self.YS.get_all_classes():
                    is_update_required=True

        # Something missing - update required
        if is_update_required:
            if classifier_type=='clipseg':
                from change_detection.clip_segmentation import clip_seg
                self.YS=clip_seg(tgt_class_list)
                self.classifier_type=classifier_type
            elif classifier_type=='yolo_world':
                from change_detection.yolo_world_segmentation import yolo_world_segmentation
                self.YS=yolo_world_segmentation(tgt_class_list)
                self.classifier_type=classifier_type
            elif classifier_type=='yolo':
                from change_detection.yolo_segmentation import yolo_segmentation
                self.YS=yolo_segmentation(tgt_class_list)
                self.classifier_type=classifier_type
            elif classifier_type=='omdet':
                from change_detection.omdet_segmentation import omdet_segmentation
                self.YS=omdet_segmentation(tgt_class_list)
                self.classifier_type=classifier_type
            elif classifier_type=='hybrid':
                from change_detection.clip_segmentation import clip_seg
                from change_detection.omdet_segmentation import omdet_segmentation
                self.YS=clip_seg(tgt_class_list)
                self.YS2=omdet_segmentation(tgt_class_list)
                self.classifier_type=classifier_type


    def process_image(self, tgt_class, detection_threshold, segmentation_save_file=None):
        # Recover the segmentation file
        if segmentation_save_file is not None and os.path.exists(segmentation_save_file):
            if not self.YS.load_file(segmentation_save_file,threshold=detection_threshold):
                return None
        else:
            # self.YS.process_image_numpy(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)    
            # This numpy bit was originally done to handle images coming from the robot ...
            #   may need to correct for live image stream processing
            self.YS.process_image(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)    
        return self.get_pts_per_class(tgt_class)
      
    def multi_prompt_process(self, prompts:list, detection_threshold, rotate90:bool=False, classifier_type='clipseg'):
        self.setup_image_processing(prompts, classifier_type)

        # Process images with both models
        if rotate90:
            rot_color = np.rot90(self.loaded_image['colorT'].cpu().numpy(), k=1, axes=(1,0))
            self.YS.process_image_numpy(rot_color, detection_threshold)   
            if classifier_type=='hybrid':
                self.YS2.process_image_numpy(rot_color, detection_threshold) 
        else:
            self.YS.process_image_numpy(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)
            if classifier_type=='hybrid':
                self.YS2.process_image_numpy(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)    
    
        all_pts = dict()
        
        # Build the class associated mask for this image and combine results
        for tgt_class in prompts:
            try:
                pts1 = self.get_pts_per_class(tgt_class, rotate90=rotate90)
                
                if classifier_type=='hybrid' and pts1 is not None:
                    pts2 = self.get_pts_per_class2(tgt_class, rotate90=rotate90)
                    
                    if pts2 is not None:
                        # Get maximum confidence scores from each model
                        max_conf1 = torch.max(pts1['probs']).item() if pts1['probs'].numel() > 0 else 0.0
                        max_conf2 = torch.max(pts2['probs']).item() if pts2['probs'].numel() > 0 else 0.0
                        
                        # Calculate dynamic weights based on relative confidence
                        total_conf = max_conf1 + max_conf2
                        if total_conf > 0:
                            # Fully dynamic weights based on confidence
                            weight1 = max_conf1 / total_conf
                            weight2 = max_conf2 / total_conf
                        else:
                            # Default to equal weights if no confidence
                            weight1 = 0.5
                            weight2 = 0.5
                        
                        # Apply weights to probabilities
                        pts1_probs = pts1['probs'] * weight1
                        pts2_probs = pts2['probs'] * weight2
                        
                        # Combine points from both models - with explicit garbage collection
                        # Filter out NaN values before combining
                        pts1_valid_mask = ~torch.isnan(pts1['xyz']).any(dim=1)
                        pts2_valid_mask = ~torch.isnan(pts2['xyz']).any(dim=1)
                        
                        if torch.sum(pts1_valid_mask) == 0 and torch.sum(pts2_valid_mask) == 0:
                            print(f"Warning: No valid points found for {tgt_class}, skipping")
                            all_pts[tgt_class] = None
                            continue
                                
                        # Filter points to remove NaNs
                        pts1_xyz_filtered = pts1['xyz'][pts1_valid_mask]
                        pts1_rgb_filtered = pts1['rgb'][pts1_valid_mask]
                        pts1_probs_filtered = pts1_probs[pts1_valid_mask]
                        
                        pts2_xyz_filtered = pts2['xyz'][pts2_valid_mask]
                        pts2_rgb_filtered = pts2['rgb'][pts2_valid_mask]
                        pts2_probs_filtered = pts2_probs[pts2_valid_mask]
                        
                        # Check if we have valid points after filtering
                        if pts1_xyz_filtered.shape[0] == 0 and pts2_xyz_filtered.shape[0] == 0:
                            print(f"Warning: All points for {tgt_class} were NaN, skipping")
                            all_pts[tgt_class] = None
                            continue
                        elif pts1_xyz_filtered.shape[0] == 0:
                            all_pts[tgt_class] = {
                                'xyz': pts2_xyz_filtered, 
                                'rgb': pts2_rgb_filtered,
                                'probs': pts2_probs_filtered
                            }
                        elif pts2_xyz_filtered.shape[0] == 0:
                            all_pts[tgt_class] = {
                                'xyz': pts1_xyz_filtered, 
                                'rgb': pts1_rgb_filtered,
                                'probs': pts1_probs_filtered
                            }
                        else:
                            # Both models have valid points, combine them
                            all_pts[tgt_class] = {
                                'xyz': torch.vstack([pts1_xyz_filtered, pts2_xyz_filtered]),
                                'rgb': torch.vstack([pts1_rgb_filtered, pts2_rgb_filtered]),
                                'probs': torch.hstack([pts1_probs_filtered, pts2_probs_filtered])
                            }
                    else:
                        all_pts[tgt_class] = pts1
                else:
                    all_pts[tgt_class] = pts1
            except Exception as e:
                print(f"Error processing class {tgt_class}: {e}")
                all_pts[tgt_class] = None
        
        return all_pts

    def process_fList(self, fList:rgbd_file_list, tgt_class, conf_threshold, classifier_type='clipseg'):
        save_fName=fList.get_combined_raw_fileName(tgt_class,classifier_type)
        pcloud=None
        if os.path.exists(save_fName):
            try:
                with open(save_fName, 'rb') as handle:
                    pcloud=pickle.load(handle)
            except Exception as e:
                pcloud=None
                print("Failed to load save file - rebuilding... " + save_fName)
        
        if pcloud is None:
            # Setup the classifier
            self.setup_image_processing([tgt_class], classifier_type)

            # Build the pcloud from individual images
            # pcloud={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8),'probs': []}
            pcloud={'xyz': torch.zeros((0,3),dtype=torch.float32,device=DEVICE),
                            'rgb': torch.zeros((0,3),dtype=torch.uint8,device=DEVICE),
                            'probs': torch.zeros((0,),dtype=torch.float,device=DEVICE)}
            count=0
            intermediate_files=[]
            deltaT=np.zeros((3,),dtype=float)
            for key in fList.keys():                
                t_array=[]
                try:
                    t_array.append(time.time())
                    self.load_image_from_file(fList, key)
                    t_array.append(time.time())
                    icloud=self.process_image(tgt_class, conf_threshold, segmentation_save_file=fList.get_segmentation_fileName(key, False, tgt_class))                    
                    t_array.append(time.time())
                    if icloud is not None and icloud['xyz'].shape[0]>100:
                        # pcloud['xyz']=np.vstack((pcloud['xyz'],icloud['xyz']))
                        # pcloud['rgb']=np.vstack((pcloud['rgb'],icloud['rgb']))
                        # pcloud['probs']=np.hstack((pcloud['probs'],icloud['probs']))
                        pcloud['xyz']=torch.vstack((pcloud['xyz'],icloud['xyz']))
                        pcloud['rgb']=torch.vstack((pcloud['rgb'],icloud['rgb']))
                        pcloud['probs']=torch.hstack((pcloud['probs'],icloud['probs']))                        
                    t_array.append(time.time())
                    deltaT=deltaT+np.diff(np.array(t_array))
                    count+=1
                    if count % 500 == 0:
                        # Save the intermediate files and clear the cache
                        fName_tmp=save_fName+"."+str(count)
                        intermediate_files.append(fName_tmp)
                        with open(fName_tmp,'wb') as handle:
                            pickle.dump(pcloud, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        deltaT2 = deltaT/count
                        print("Time Array")
                        print(f" -- Loading    {deltaT2[0]}")
                        print(f" -- Processing {deltaT2[1]}")
                        print(f" -- np.vstack  {deltaT2[2]}")
                        pcloud={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8),'probs': []}            
                except Exception as e:
                    print("Image not loaded - " + str(e))
            for f in intermediate_files:
                with open(f, 'rb') as handle:
                    pcloud_tmp=pickle.load(handle)
                pcloud['xyz']=torch.vstack((pcloud['xyz'],pcloud_tmp['xyz']))
                pcloud['rgb']=torch.vstack((pcloud['rgb'],pcloud_tmp['rgb']))
                pcloud['probs']=torch.hstack((pcloud['probs'],pcloud_tmp['probs']))                        
                os.remove(f)

            pcloud['xyz']=pcloud['xyz'].cpu().numpy()
            pcloud['rgb']=pcloud['rgb'].cpu().numpy()
            pcloud['probs']=pcloud['probs'].cpu().numpy()
            # pdb.set_trace()
            # Now save the result so we don't have to keep processing this same cloud
            with open(save_fName,'wb') as handle:
                pickle.dump(pcloud, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Finally - filter the cloud with the requested confidence threshold
        whichP=(pcloud['probs']>conf_threshold)
        print(pcloud)
        return {'xyz':pcloud['xyz'][whichP],'rgb':pcloud['rgb'][whichP],'probs':pcloud['probs'][whichP]}

    def process_fList_multi(self, fList:rgbd_file_list, tgt_class_list:list, conf_threshold, classifier_type='clipseg'):
        save_fName=dict()
        pcloud=dict()
        intermediate_files=dict()
        for tgt in tgt_class_list:
            save_fName[tgt]=fList.get_combined_raw_fileName(tgt,classifier_type)
            # Only going to track point clouds for files that do not already exist...
            if not os.path.exists(save_fName[tgt]):
                pcloud[tgt]={'xyz': torch.zeros((0,3),dtype=torch.float32,device=DEVICE),
                             'rgb': torch.zeros((0,3),dtype=torch.uint8,device=DEVICE),
                             'probs': torch.zeros((0,),dtype=torch.float,device=DEVICE)}
                intermediate_files[tgt]=[]

        self.setup_image_processing(tgt_class_list, classifier_type)

        # Build the pcloud from individual images
        count=0
        deltaT=np.zeros((3,),dtype=float)
        for key in fList.keys():                
            t_array=[]
            try:
                t_array.append(time.time())
                self.load_image_from_file(fList, key)
                t_array.append(time.time())
                icloud=self.multi_prompt_process(tgt_class_list, conf_threshold, classifier_type=classifier_type)
                t_array.append(time.time())
                for tgt in icloud.keys():
                    if icloud[tgt] is not None:
                        if tgt in pcloud and icloud[tgt]['xyz'].shape[0]>100:
                            pcloud[tgt]['xyz']=torch.vstack((pcloud[tgt]['xyz'],icloud[tgt]['xyz']))
                            pcloud[tgt]['rgb']=torch.vstack((pcloud[tgt]['rgb'],icloud[tgt]['rgb']))
                            pcloud[tgt]['probs']=torch.hstack((pcloud[tgt]['probs'],icloud[tgt]['probs']))
                t_array.append(time.time())
                deltaT=deltaT+np.diff(np.array(t_array))
                count+=1
                if count % 500 == 0:
                    # Save the intermediate files and clear the cache
                    for tgt in pcloud.keys():
                        # Is this file basically empty? Then don't bother saving
                        if pcloud[tgt]['xyz'].shape[0]>100:
                            fName_tmp=save_fName[tgt]+"."+str(count)
                            intermediate_files[tgt].append(fName_tmp)
                            with open(fName_tmp,'wb') as handle:
                                pickle.dump(pcloud[tgt], handle, protocol=pickle.HIGHEST_PROTOCOL)
                            pcloud[tgt]={'xyz': torch.zeros((0,3),dtype=torch.float32,device=DEVICE),
                                'rgb': torch.zeros((0,3),dtype=torch.uint8,device=DEVICE),
                                'probs': torch.zeros((0,),dtype=torch.float,device=DEVICE)}      
                    deltaT2 = deltaT/count
                    print("Time Array")
                    print(f" -- Loading    {deltaT2[0]}")
                    print(f" -- Processing {deltaT2[1]}")
                    print(f" -- np.vstack  {deltaT2[2]}")
            except Exception as e:
                print("Image not loaded - " + str(e))
        
        # All files processed - now combine the intermediate results and generate a single cloud for each
        #   target object type
        for tgt in pcloud.keys():
            for f in intermediate_files[tgt]:
                with open(f, 'rb') as handle:
                    pcloud_tmp=pickle.load(handle)
                pcloud[tgt]['xyz']=torch.vstack((pcloud[tgt]['xyz'],pcloud_tmp['xyz']))
                pcloud[tgt]['rgb']=torch.vstack((pcloud[tgt]['rgb'],pcloud_tmp['rgb']))
                pcloud[tgt]['probs']=torch.hstack((pcloud[tgt]['probs'],pcloud_tmp['probs']))
                os.remove(f)

            pcloud_np={'xyz': pcloud[tgt]['xyz'].cpu().numpy(), 
                       'rgb': pcloud[tgt]['rgb'].cpu().numpy(), 
                       'probs': pcloud[tgt]['probs'].cpu().numpy(), 
                       }
            print(pcloud_np)
            # Now save the result so we don't have to keep processing this same cloud
            with open(save_fName[tgt],'wb') as handle:
                pickle.dump(pcloud_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # And clear the final point cloud
            pcloud[tgt]={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8),'probs': []} 

TIME_STRUCT={'count': 0, 'times':np.zeros((3,),dtype=float)}

def get_distinct_clusters(pcloud, gridcell_size=DBSCAN_GRIDCELL_SIZE, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, cluster_min_count=CLUSTER_MIN_COUNT, floor_threshold=0.1):
    global TIME_STRUCT
    t_array=[]
    t_array.append(time.time())

    clouds=[]
    if pcloud is None or len(pcloud.points)<cluster_min_count:
        return clouds
    if gridcell_size>0:
        pcd_small=pcloud.voxel_down_sample(gridcell_size)
        t_array.append(time.time())
        pts=np.array(pcd_small.points)
        p2=DBSCAN(eps=eps, min_samples=min_samples,n_jobs=10).fit(pts)
        t_array.append(time.time())
    else:
        pts=np.array(pcloud.points)
        p2=DBSCAN(eps=eps, min_samples=min_samples,n_jobs=10).fit(pts)

    # Need to get the cluster sizes... so we can focus on the largest clusters only
    cl_cnt=np.array([ (p2.labels_==cnt).sum() for cnt in range(p2.labels_.max() + 1) ])
    validID=np.where(cl_cnt>cluster_min_count)[0]
    if validID.shape[0]>0:
        sortedI=np.argsort(-cl_cnt[validID])

        for id in validID[sortedI][:10]:
            whichP=(p2.labels_==id)
            pts2=pts[whichP]
            whichP2=(pts2[:,2]>floor_threshold)
            if whichP2.sum()>cluster_min_count:
                clouds.append(object_pcloud(pts2[whichP2], sample=False))
    t_array.append(time.time())
    TIME_STRUCT['times']=TIME_STRUCT['times']+np.diff(np.array(t_array))
    TIME_STRUCT['count']+=1
    if TIME_STRUCT['count']%100==0:
        print(f"******* TIME(get_distinct_clusters) - {TIME_STRUCT['count']} ****** ")
        print(TIME_STRUCT['times']/500)
        TIME_STRUCT['times']=np.zeros((3,),dtype=float)
        # pdb.set_trace()

    return clouds

class object_pcloud():
    def __init__(self, pts, label:str=None, num_samples=1000, sample=True):
        self.box=np.vstack((pts.min(0),pts.max(0)))
        self.pts=pts
        self.pts_shape=self.pts.shape
        self.label=label
        self.farthestP=None
        if sample:
            self.sample_pcloud(num_samples)
        self.prob_stats=None
        self.centroid=self.pts.mean(0)

    def sample_pcloud(self, num_samples):
            self.farthestP=farthest_point_sampling(self.pts, num_samples)

    def set_label(self, label):
        self.label=label
    
    def is_box_overlap(self, input_cloud, dimensions=[0,1,2], threshold=0.3):
        for dim in dimensions:
            if self.box[1,dim]<(input_cloud.box[0,dim]-threshold) or self.box[0,dim]>=(input_cloud.box[1,dim]+threshold):
                return False
        return True

    def compute_cloud_distance(self, input_cloud):
        input_pt_matrix=input_cloud.pts[input_cloud.farthestP]
        min_sq_dist=1e10
        for pid in self.farthestP:
            min_sq_dist=min(min_sq_dist, ((input_pt_matrix-self.pts[pid])**2).sum(1).min())
        return np.sqrt(min_sq_dist)
    
    def is_above(self, input_cloud):
        # Should be overlapped in x + y directions
        if self.centroid[0]>input_cloud.box[0,0] and self.centroid[0]<input_cloud.box[1,0] and self.centroid[1]>input_cloud.box[0,1] and self.centroid[1]<input_cloud.box[1,1]:
            # Should also be "above" the other centroid
            return self.centroid[2]>input_cloud.centroid[2]
        return False
    
    def estimate_probability(self, original_xyz, original_prob):
        filt=(original_xyz[:,0]>=self.box[0][0])*(original_xyz[:,0]<=self.box[1][0])*(original_xyz[:,1]>=self.box[0][1])*(original_xyz[:,1]<=self.box[1][1])*(original_xyz[:,2]>=self.box[0][2])*(original_xyz[:,2]<=self.box[1][2])
        self.prob_stats=dict()
        self.prob_stats['max']=original_prob[filt].max()
        self.prob_stats['mean']=original_prob[filt].mean()
        self.prob_stats['stdev']=original_prob[filt].std()
        self.prob_stats['pcount']=filt.shape[0]
    
    def size(self):
        return self.pts_shape[0]
    
    def compress_object(self):
        self.pts=None
        self.farthestP=None