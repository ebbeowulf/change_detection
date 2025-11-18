# Common utilities for creating point clouds from RGB-D images
#   Includes:
#   - pcloud_base: parent class with common utilities for point cloud creation
#   - pcloud_openVocab: child class for open-vocabulary point cloud creation
#   - build_dbscan_boxes: utility function to build boxes around clusters in 2D images
#   - box_iou: utility function to compute intersection-over-union for boxes
#   - is_box_overlap: utility function to check for box overlap
#   - estimate_blur: utility function to estimate image blur

import torch
from change_pcloud_utils.rgbd_file_list import rgbd_file_list
import numpy as np
import cv2
from change_pcloud_utils.map_utils import get_rotated_points, DEVICE, connected_components_filter, get_center_point
import os
from change_pcloud_utils.camera_params import camera_params
import pickle
import time
from PIL import Image

DEPTH_BLUR_THRESHOLD=None #Applied to Depth images
COLOR_BLUR_THRESHOLD=None #Applied to Color images

#Parent class containing common utilities for creating point clouds
#   load_image / load_image_from_file - stores images internally for futher processing
#   get_pts - recovers rotated points from the stored files given a prepared mask
#   setup_image_processing - setup an open-vocabulary processing method
class pcloud_base():
    def __init__(self, params:camera_params):
        self.params=params
        self.YS=None
        self.rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
        self.cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)
        self.rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)        
        self.loaded_image=None
        self.classifier_type=None

    # Image loading to allow us to process more than one class in rapid succession
    # def load_image_from_file(self, fList:rgbd_file_list, image_key, max_distance=10.0):
    #     colorI=cv2.imread(fList.get_color_fileName(image_key), -1)
    #     colorI=Image.open(fList.get_color_fileName(image_key))
    #     depthI=cv2.imread(fList.get_depth_fileName(image_key), -1)
    #     poseM=fList.get_pose(image_key)
    #     self.load_image(colorI, depthI, poseM, image_key, max_distance=max_distance)

    # Load an image into the internal memory structure for further processing
    def load_image(self, 
                   colorI_PIL:Image, #assumes PIL format
                   depthI_np:np.ndarray, 
                   poseM:np.ndarray, 
                   uid_key:str, 
                   max_distance=10.0,
                   color_blur_threshold=None,
                   depth_blur_threshold=None):
        if self.loaded_image is None or self.loaded_image['key']!=uid_key:
            try:
                if self.loaded_image is None:
                    self.loaded_image=dict()
                self.loaded_image['color']=colorI_PIL
                self.loaded_image['depthT']=torch.tensor(depthI_np.astype('float')/1000.0,device=DEVICE)
                self.loaded_image['colorT']=torch.tensor(np.array(colorI_PIL),device=DEVICE)
                self.loaded_image['x'] = self.cols*self.loaded_image['depthT']/self.params.fx
                self.loaded_image['y'] = self.rows*self.loaded_image['depthT']/self.params.fy
                self.loaded_image['depth_mask']=(self.loaded_image['depthT']>1e-4)*(self.loaded_image['depthT']<max_distance)
                if depth_blur_threshold is not None:
                    blur=torch.tensor(estimate_blur(depthI_np),device=DEVICE)
                    self.loaded_image['depth_mask']*=(blur>depth_blur_threshold)
                if color_blur_threshold is not None:
                    blur=torch.tensor(estimate_blur(np.array(colorI_PIL)),device=DEVICE)
                    self.loaded_image['depth_mask']*=(blur>color_blur_threshold)                    

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

    # Return a set of XYZRGB points and associated probabilities from the images given a mask image
    def get_pts(self, prob_array, filtered_maskT_bool, rotate90=False):
        # Return all points associated with the target class
        pts_rot=get_rotated_points(self.loaded_image['x'],self.loaded_image['y'],self.loaded_image['depthT'],filtered_maskT_bool,self.loaded_image['M']) 
        if rotate90:
            probs=torch.rot90(prob_array,dims=(0,1))
            return {'xyz': pts_rot, 
                    'rgb': self.loaded_image['colorT'][filtered_maskT_bool], 
                    'probs': probs[filtered_maskT_bool]}
        else:
            return {'xyz': pts_rot, 
                    'rgb': self.loaded_image['colorT'][filtered_maskT_bool], 
                    'probs': prob_array[filtered_maskT_bool]}

    def sam_segmentation(self, xy_points:list):
        from transformers import SamModel, SamProcessor
        if not hasattr(self, 'sam_model'):
            self.sam_model=SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
        if not hasattr(self, 'sam_processor'):
            self.sam_processor=SamProcessor.from_pretrained("facebook/sam-vit-huge")
        inputs = self.sam_processor(self.loaded_image['color'], input_points=[xy_points], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        masks=self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        return masks[0].data[0][0].cpu().numpy()

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
                from segmentation_utils.clip_segmentation import clip_seg
                self.YS=clip_seg(tgt_class_list)
                self.classifier_type=classifier_type
            elif classifier_type=='yolo_world':
                from segmentation_utils.yolo_world_segmentation import yolo_world_segmentation
                self.YS=yolo_world_segmentation(tgt_class_list)
                self.classifier_type=classifier_type
            elif classifier_type=='yolo':
                from segmentation_utils.yolo_segmentation import yolo_segmentation
                self.YS=yolo_segmentation(tgt_class_list)
                self.classifier_type=classifier_type
    
    def generate_boxes_with_sam(self,delta_np,filtered_mask_np,generate_mask=False,merge_overlap=False):
        # First step ... need some seed boxes. Going to use dbscan for this
        dbscan_boxes=build_dbscan_boxes(delta_np,filtered_mask_np)

        if len(dbscan_boxes)==0:
            return []

        # Second step... refinement with SAM
        #    for each dbscan box, generate a SAM mask.
        #    Only keep those that have significant overlap - using a fixed iou threshold for now        
        mask_combo=None
        sam_boxes=[]
        for box in dbscan_boxes:
            # Sample points from the box - but move in 10% from the margins
            #   to avoid sampling outside the object of interest
            box_dim=np.array(box[1][2:])-np.array(box[1][:2])
            deltaDim=(box_dim/20).astype(int)
            new_box=np.hstack((np.array(box[1][:2])+deltaDim,np.array(box[1][2:])-deltaDim))
            subR=filtered_mask_np[new_box[1]:new_box[3],new_box[0]:new_box[2]]
            rowD,colD=np.nonzero(subR)
            whichP=np.random.choice(np.arange(rowD.shape[0]),10)
            xy_points=np.vstack((colD[whichP]+box[1][0],rowD[whichP]+box[1][1])).transpose().tolist()

            # Now run the segmentation step
            sam_mask=self.sam_segmentation(xy_points)

            if generate_mask: #for now, all masks are merged into a single mask
                if mask_combo is None:
                    mask_combo=sam_mask
                else:
                    mask_combo*=sam_mask
            
            # Identify the boundaries of the segmented zone
            rowS,colS=np.nonzero(sam_mask)
            sbox=[colS.min(),rowS.min(),colS.max(),rowS.max()]
            if box_iou(box[1],sbox)>0.3:
                max_prob=0
                for cbox in dbscan_boxes:
                    if is_box_overlap(sbox,cbox[1]):
                        max_prob=max((max_prob,cbox[0]))
                sam_boxes.append((max_prob,sbox))

        # Do we merge overlapping boxes?
        if merge_overlap:
            count_boxes=len(sam_boxes)+1
            while len(sam_boxes)>1 and count_boxes>len(sam_boxes):
                count_boxes=len(sam_boxes)
                old_boxes=sam_boxes
                sam_boxes=[]
                is_available=np.ones((len(old_boxes)),dtype=bool)
                for box_idx, box in enumerate(old_boxes):
                    if not is_available[box_idx]:
                        continue
                    is_available[box_idx]=False
                    tgt_box=box
                    for box_idx2 in range(box_idx+1,len(old_boxes)):
                        if is_box_overlap(tgt_box[1],old_boxes[box_idx2][1]):
                            is_available[box_idx2]=False
                            combo=np.vstack((tgt_box[1],old_boxes[box_idx2][1]))
                            tgt_box=(max(tgt_box[0],old_boxes[box_idx2][0]),
                                np.hstack((combo[:,:2].min(0),combo[:,2:].max(0))))
                    sam_boxes.append(tgt_box)
        if generate_mask:
            return sam_boxes, mask_combo            
        else:
            return sam_boxes
        
# Child class for open vocabulary image processing
#   After loading an image, call process_image or process_image_multi to get all points
#   associated with each of the specified prompts
class pcloud_openVocab(pcloud_base):
    def __init__(self, params:camera_params):
        self.params=params
        self.YS=None
        self.rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
        self.cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)
        self.rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)        
        self.loaded_image=None
        self.classifier_type=None
    
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
            
            return self.get_pts(self.YS.get_prob_array(tgt_class),filtered_maskT,rotate90)
        else:
            return None
        
    # Process an image with a single prompt, saving the resulting points
    def process_image(self, tgt_class, detection_threshold, segmentation_save_file=None):
        # Recover the segmentation file
        if segmentation_save_file is not None and os.path.exists(segmentation_save_file):
            if not self.YS.load_file(segmentation_save_file,threshold=detection_threshold):
                return None
        else:
            # self.YS.process_image_numpy(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)    
            # This numpy bit was originally done to handle images coming from the robot ...
            #   may need to correct for live image stream processing
            #self.YS.process_image(self.loaded_image['colorT'].cpu().numpy(), detection_threshold)    
            self.YS.process_image(self.loaded_image['color'].cpu().numpy(), detection_threshold)    
        return self.get_pts_per_class(tgt_class)
      
    # Process all images in the fList with a single prompt - probably broken after changing all loaded color files to PIL format
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
        return {'xyz':pcloud['xyz'][whichP],'rgb':pcloud['rgb'][whichP],'probs':pcloud['probs'][whichP]}

    # Process an image with multiple prompts, saving the resulting points
    def multi_prompt_process(self, prompts:list, detection_threshold, rotate90:bool=False, classifier_type='clipseg'):
        self.setup_image_processing(prompts,classifier_type)

        if rotate90:
            rot_color=np.rot90(self.loaded_image['colorT'].cpu().numpy(), k=1, axes=(1,0))
            self.YS.process_image_numpy(rot_color, detection_threshold)    
        else:
            # self.YS.process_image_numpy(self.loaded_image['color'], detection_threshold)    
            self.YS.process_image(self.loaded_image['color'], detection_threshold)    

        all_pts=dict()
        # Build the class associated mask for this image
        for tgt_class in prompts:
            all_pts[tgt_class]=self.get_pts_per_class(tgt_class, rotate90=rotate90)

        return all_pts
    
    # Process all images in the fList with multiple prompts
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
            # Now save the result so we don't have to keep processing this same cloud
            with open(save_fName[tgt],'wb') as handle:
                pickle.dump(pcloud_np, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # And clear the final point cloud
            pcloud[tgt]={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8),'probs': []} 

# Child class for change detection processing
class pcloud_change(pcloud_base):
    def __init__(self, params:camera_params):
        self.params=params
        self.YS=None
        self.rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
        self.cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)
        self.rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)        
        self.loaded_image=None
        self.classifier_type=None

    def openvocab_segmentation(self, 
                      prompts,
                      baseline_image): 
        latest_result=dict()
        for query in prompts:
            latest_result[query]={'new_prob': None, 'baseline_prob': None}
        
        self.YS.process_image(self.loaded_image['color'])
        for query in prompts:
            latest_result[query]['new_prob'] = self.YS.get_prob_array(query)

        self.YS.process_image(baseline_image)
        for query in prompts:
            latest_result[query]['baseline_prob'] = self.YS.get_prob_array(query)
        
        return latest_result

    def multi_prompt_change_process(self, 
                             baseline_image, # color image of same vantage point generated by nerfstudio
                             prompts:list, 
                             detection_threshold:float, 
                             classifier_type:str='clipseg',
                             is_positive_change=True,
                             est_bboxes=False):
        self.setup_image_processing(prompts,classifier_type)
        latest_result = self.openvocab_segmentation(prompts, baseline_image)

        all_pts=dict()
        all_bboxes=dict()
        # Build the class associated mask for this image - discard any images where the counts are too high as likely 
        #       localization or generative errors. If "change" > 10%, then probably couldn't localize the image very well anyways
        max_point_count=self.loaded_image['colorT'].shape[0]*self.loaded_image['colorT'].shape[1]/10
        for tgt_class in prompts:
            if tgt_class in latest_result and latest_result[tgt_class]['new_prob'] is not None and latest_result[tgt_class]['baseline_prob'] is not None:
                # Get the change image by subtracting clipseg results
                if is_positive_change:
                    deltaT=latest_result[tgt_class]['new_prob']-latest_result[tgt_class]['baseline_prob']
                else:
                    deltaT=latest_result[tgt_class]['baseline_prob']-latest_result[tgt_class]['new_prob']

                maskT=(deltaT>detection_threshold)
                num_change_points=maskT.sum()
                print(f"num change points = {num_change_points}")
                if num_change_points>0 and num_change_points<max_point_count:
                    filtered_maskT=self.loaded_image['depth_mask']*maskT
                    all_pts[tgt_class]=self.get_pts(deltaT, filtered_maskT)
                    if est_bboxes:                                                
                        all_bboxes[tgt_class]=self.generate_boxes_with_sam(deltaT.cpu().numpy(),filtered_maskT.cpu().numpy())

                        if 0: # change to draw the images and masks
                            im_out=np.array(self.loaded_image['color'])
                            im_out[:,:,0][filtered_maskT.cpu().numpy()]=255
                            for bbox in all_bboxes[tgt_class]:
                                im_out=cv2.rectangle(im_out, bbox[1][:2], bbox[1][2:], (0,0,255), 2)
                            cv2.imshow("mask",im_out[:,:,[2,1,0]])
                            cv2.waitKey(1)                        
        
        if est_bboxes:
            return all_pts, all_bboxes
        return all_pts

def is_box_overlap(bbox1,bbox2):
    from shapely import Polygon
    polygon1 = Polygon([bbox1[:2], [bbox1[2],bbox1[1]], bbox1[2:],[bbox1[0],bbox1[3]]])
    polygon2 = Polygon([bbox2[:2], [bbox2[2],bbox2[1]], bbox2[2:],[bbox2[0],bbox2[3]]])
    return polygon1.intersects(polygon2)

def box_iou(bbox1,bbox2):
    from shapely import Polygon
    polygon1 = Polygon([bbox1[:2], [bbox1[2],bbox1[1]], bbox1[2:],[bbox1[0],bbox1[3]]])
    polygon2 = Polygon([bbox2[:2], [bbox2[2],bbox2[1]], bbox2[2:],[bbox2[0],bbox2[3]]])
    if polygon1.intersects(polygon2):
        p_area1=polygon1.area
        p_area2=polygon2.area
        i_area=polygon1.intersection(polygon2).area
        return i_area/(p_area1+p_area2-i_area)
    return 0.0

def build_dbscan_boxes(prob_image, mask, eps=10, min_samples=20, MAX_CLUSTERING_SAMPLES=50000):
    from sklearn.cluster import DBSCAN

    rows,cols=np.nonzero(mask)
    xy_grid_pts=np.vstack((cols,rows)).transpose()
    scores=prob_image[rows,cols]
    if xy_grid_pts is None or xy_grid_pts.shape[0]<min_samples:
        return []

    # Need to constrain the maximum number of points - else dbscan will be extremely slow
    if xy_grid_pts.shape[0]>MAX_CLUSTERING_SAMPLES:        
        rr=np.random.choice(np.arange(xy_grid_pts.shape[0]),size=MAX_CLUSTERING_SAMPLES)
        xy_grid_pts=xy_grid_pts[rr]
        scores=scores[rr]

    CL2=DBSCAN(eps=eps, min_samples=min_samples).fit(xy_grid_pts,sample_weight=scores)
    boxes=[]
    for idx in range(10):
        whichP=np.where(CL2.labels_== idx)            
        if len(whichP[0])<1:
            break
        box=np.hstack((xy_grid_pts[whichP].min(0),xy_grid_pts[whichP].max(0)))
        boxes.append((scores[whichP].max(),box))
    return boxes

def estimate_blur(gray_np,step=5,filter_size=50):
    laplacian=cv2.Laplacian(gray_np, cv2.CV_64F)
    max_row=laplacian.shape[0]-filter_size
    max_col=laplacian.shape[1]-filter_size
    Lpl_var=np.zeros((int(np.floor(max_row/step)),int(np.floor(max_col/step))),dtype=float)
    for var_row,row in enumerate(range(0,laplacian.shape[0]-50,5)):
        for var_col, col in enumerate(range(0,laplacian.shape[1]-50,5)):
            Lpl_var[var_row,var_col]=laplacian[row:(row+filter_size),col:(col+filter_size)].var()
    blur=cv2.resize(Lpl_var,(gray_np.shape[1],gray_np.shape[0]))
    return blur

# Build change detection point clouds from two rgbd_file_lists
#    used for both phone and robot captured images
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
        
        if(pcloud_creator.load_image(colorI_new, depthI, M, str(key),color_blur_threshold=COLOR_BLUR_THRESHOLD, depth_blur_threshold=DEPTH_BLUR_THRESHOLD)):
            results=pcloud_creator.multi_prompt_process(prompts, det_threshold)
            for query in prompts:
                if query in results and results[query]['xyz'].shape[0]>0:
                    pcloud[query]['xyz']=torch.vstack((pcloud[query]['xyz'],results[query]['xyz']))
                    pcloud[query]['probs']=torch.hstack((pcloud[query]['probs'],results[query]['probs']))
                    pcloud[query]['rgb']=torch.vstack((pcloud[query]['rgb'],results[query]['rgb']))
        else:
            print(f"Skipping image {key} - not loaded properly")
        
    return pcloud

def build_pclouds(fList_new:rgbd_file_list,
                  fList_renders:rgbd_file_list,
                  prompts:list,
                  params:camera_params,
                  detection_threshold:float,
                  rebuild_pcloud:bool=False):
    # build clouds if necessary - return list of filenames for saved pclouds
    pcloud_fNames=dict()
    all_files_exist=True
    for key in prompts:
        P1=key.replace(' ','_')
        if fList_renders is not None:
            pcloud_fNames[key]=f"{fList_new.intermediate_save_dir}/{P1}.{detection_threshold}.pcloud.pkl"
        else:
            pcloud_fNames[key]=f"{fList_new.intermediate_save_dir}/{P1}.{detection_threshold}.OV.pcloud.pkl"
        
        # Does the file exist already?
        if not os.path.exists(pcloud_fNames[key]):
            all_files_exist=False

    # Rebuild pclouds if requested 
    if not all_files_exist or rebuild_pcloud:
        if fList_renders is not None:
            # Do the original change detection experiment
            pcloud=build_change_clouds(params, 
                                    fList_new, 
                                    fList_renders, 
                                    prompts, 
                                    detection_threshold)            
        else:
            # Use open vocabulary models only - no change applied
            pcloud=build_openVocab_clouds(params, 
                                    fList_new, 
                                    prompts, 
                                    detection_threshold)
        # Save the result
        for key in pcloud:
            with open(pcloud_fNames[key],'wb') as handle:
                pickle.dump(pcloud[key], handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
    # Return the list of files to be loaded
    return pcloud_fNames

if __name__ == '__main__':
    import argparse
    from colmap_utils import get_camera_params, build_file_list

    parser = argparse.ArgumentParser()
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('root_dir',type=str,help='root project folder where the images and colmap info are stored')
    parser.add_argument('prompt',type=str,help='what is the prompt to search for in the images?')
    parser.add_argument('--color_dir',type=str,default='images_combined',help='where are the color images? (default=images_combined)')
    parser.add_argument('--depth_dir',type=str,default='depth',help='where are the color images? (default=depth)')
    parser.add_argument('--colmap_dir',type=str,default='colmap_combined/sparse_geo',help='where are the images + cameras.txt files? (default=colmap_combined/sparse)')
    parser.add_argument('--save_dir',type=str,default='pclouds',help='where to save the point clouds? (default=pclouds)')
    parser.add_argument('--detection_threshold',type=float,default=0.5,help='detection threshold for point cloud generation (default=0.5)')
    parser.add_argument('--frame_keyword',type=str,default='new',help='keyword to identify frames in colmap (default=new)')
    args=parser.parse_args()

    save_dir=f"{args.root_dir}/{args.save_dir}/"
    color_image_dir=f"{args.root_dir}/{args.color_dir}/"
    rendered_image_dir=f"{args.root_dir}/{args.depth_dir}/"
    colmap_dir=f"{args.root_dir}/{args.colmap_dir}/"
    params=get_camera_params(colmap_dir,args.nerfacto_dir)
    fList_new=build_file_list(color_image_dir,rendered_image_dir,save_dir,colmap_dir,args.frame_keyword)

    pcloud_fNames = build_pclouds(fList_new,
                  None,
                  [args.prompt],
                  params,
                  detection_threshold=args.detection_threshold,
                  rebuild_pcloud=True)
    
    print("Generated point cloud files:")
    print(pcloud_fNames)