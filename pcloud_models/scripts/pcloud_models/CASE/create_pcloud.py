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
from change_detection.camera_params import camera_params
import copy
from map_utils import clip_threshold_evaluation, get_center_point, connected_components_filter, get_rotated_points

DEVICE = torch.device("cpu")

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
                                pdb.set_trace()
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
