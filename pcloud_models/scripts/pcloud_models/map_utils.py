import torch
import pickle
import open3d as o3d
import numpy as np
import cv2
from change_detection.yolo_segmentation import yolo_segmentation
import os

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

class camera_params():
    def __init__(self, height, width, fx, fy, cx, cy, rot_matrix):
        self.height=height
        self.width=width
        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=cy
        self.rot_matrix=rot_matrix
        self.K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

class rgbd_file_list():
    def __init__(self, color_image_dir:str, depth_image_dir:str, intermediate_save_dir:str):
        self.all_files=dict()
        self.color_image_dir=color_image_dir + "/"
        self.depth_image_dir=depth_image_dir + "/"
        self.intermediate_save_dir=intermediate_save_dir + "/"
    
    def add_file(self, id:int, color_fName:str, depth_fName:str):
        if id in self.all_files:
            print("ID %d is already in the file list - replacing existing entry")

        self.all_files[id]={'color': color_fName, 'depth': depth_fName}

    def add_pose(self, id:int, pose:np.array):
        self.all_files[id]['pose']=pose

    def keys(self):
        return self.all_files.keys()
    
    def is_key(self, id:int):
        return id in self.all_files
    
    def get_color_fileName(self, id:int):
        return self.color_image_dir+self.all_files[id]['color']

    def get_yolo_fileName(self, id:int):
        return self.intermediate_save_dir+self.all_files[id]['color']+".yolo.pkl"
    
    def get_depth_fileName(self, id:int):
        return self.depth_image_dir+self.all_files[id]['depth']
    
    def get_pose(self, id:int):
        return self.all_files[id]['pose']
    
    def get_class_pcloud_fileName(self, id:int, cls:str):
        return self.intermediate_save_dir+self.all_files[id]['color']+".%s.pkl"%(cls)


# Process all images with yolo - creating
#   pickle files for the results and storing alongside the 
#   original color images
def process_images_with_yolo(fList:rgbd_file_list):
    print("process_images")
    YS=yolo_segmentation()
    for key in fList.keys():
        print(fList.all_files[key]['colorFile'])
        pkl=fList.get_yolo_fileName(key)
        if not os.path.exists(pkl):
            img,results=YS.process_file(fList.get_color_fileName(key))
            with open(pkl, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create a list of all of the objects recognized by yolo
#   across all files. Will only load existing pkl files, not 
#   create any new ones
def create_object_list(fList:rgbd_file_list):
    YS=yolo_segmentation()
    obj_list=dict()
    for key in fList.keys():
        with open(fList.get_yolo_fileName(key), 'rb') as handle:
            results=pickle.load(handle)
            YS.load_prior_results(results)
        for id, prob in zip(YS.cl_labelID, YS.cl_probs):
            if YS.id2label[id] not in obj_list:
                obj_list[YS.id2label[id]]={'images': [key], 'probs': [prob]}
            else:
                obj_list[YS.id2label[id]]['images'].append(key)
                obj_list[YS.id2label[id]]['probs'].append(prob)
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
    pcd=o3d.geometry.PointCloud()
    if xyz_points.shape[0]<max_num_points:
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.colors = o3d.utility.Vector3dVector(rgb_points[:,[2,1,0]]/255) 
    else:
        rr=np.random.choice(np.arange(xyz_points.shape[0]),max_num_points)
        pcd.points = o3d.utility.Vector3dVector(xyz_points[rr,:])
        rgb2=rgb_points[rr,:]
        pcd.colors = o3d.utility.Vector3dVector(rgb2[:,[2,1,0]]/255) 

    return pcd

# Combine together multiple point clouds into a single
#   cloud and display the result using open3d
def visualize_combined_xyzrgb(fList:rgbd_file_list, params:camera_params, howmany_files=100):
    rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy)
    cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx)

    combined_xyz=np.zeros((0,3),dtype=float)
    combined_rgb=np.zeros((0,3),dtype=np.uint8)

    count=0
    for key in range(max(fList.keys())):
        if not fList.is_key(key):
            continue

        # Create the generic depth data
        colorI=cv2.imread(fList.get_color_fileName(key), -1)
        depthI=cv2.imread(fList.get_depth_fileName(key), -1)
        depthT=torch.tensor(depthI.astype('float')/1000.0)
        colorT=torch.tensor(colorI)
        x = cols*depthT/params.fx
        y = rows*depthT/params.fy
        depth_mask=(depthT>1e-4)*(depthT<10.0)

        # Rotate the points into the right space
        M=torch.matmul(params.rot_matrix,torch.tensor(fList.get_pose(key)))
        pts=torch.stack([x[depth_mask],y[depth_mask],depthT[depth_mask],torch.ones(((depth_mask>0).sum()))],dim=1)
        pts_rot=torch.matmul(M,pts.transpose(0,1))
        pts_rot=pts_rot[:3,:].transpose(0,1)

        if pts_rot.shape[0]>100:
            combined_xyz=np.vstack((combined_xyz,pts_rot.cpu().numpy()))
            combined_rgb=np.vstack((combined_rgb,colorT[depth_mask].cpu().numpy()))
        count+=1
        if count==howmany_files:
            break
    pcd=pointcloud_open3d(combined_xyz,rgb_points=combined_rgb)
    o3d.visualization.draw_geometries([pcd])

    return pcd

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
def get_center_point(depthT:torch.tensor, combo_mask:torch.tensor, xy_bbox, neighborhood=5):
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
    whichD=dist.argmin()
    return (indices[whichD].cpu()+torch.tensor([minR,minC])).tolist()

# Create pclouds for all of the indicated target classes, saving the resulting cloud to the
#   disk so that it does not need to be re-calculated each time. Note that the connected components
#   filter is being applied each time.
def create_pclouds(tgt_classes:list, fList:rgbd_file_list, params:camera_params, conf_threshold=0.5):
    YS=yolo_segmentation()
    rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
    cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)

    pclouds=dict()
    for cls in tgt_classes:
        pclouds[cls]={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8)}

    rot_matrixT=torch.tensor(params['rot_matrix'],device=DEVICE)
    for key in range(max(fList.keys())):
        if not fList.is_key(key):
            continue
        print(key)
        try:
            with open(fList.get_yolo_fileName(key), 'rb') as handle:
                results=pickle.load(handle)
                YS.load_prior_results(results)
            colorI=cv2.imread(fList.get_color_fileName(key), -1)
            depthI=cv2.imread(fList.get_depth_fileName(key), -1)
            depthT=torch.tensor(depthI.astype('float')/1000.0,device=DEVICE)
            colorT=torch.tensor(colorI,device=DEVICE)
            x = cols*depthT/params.fx
            y = rows*depthT/params.fy
            depth_mask=(depthT>1e-4)*(depthT<10.0)

            # Build the rotation matrix
            M=torch.matmul(rot_matrixT,torch.tensor(fList.get_pose(key),device=DEVICE))

            # Now extract a mask per category
            for cls in tgt_classes:
                cls_mask=YS.get_mask(cls)
                if cls_mask is not None:
                    save_fName=fList.get_class_pcloud_fileName(key,cls)
                    # Load or create the connected components mask for this object type
                    try:
                        if os.path.exists(save_fName):
                            with open(save_fName, 'rb') as handle:
                                filtered_maskT=pickle.load(handle)
                        else:
                            combo_mask=(torch.tensor(cls_mask,device=DEVICE)>conf_threshold)*depth_mask
                            # Find the list of boxes associated with this object
                            which_boxes=np.where(np.array(YS.cl_labelID)==YS.label2id[cls])[0]
                            filtered_maskT=None
                            for box_idx in range(which_boxes.shape[0]):
                                # Pick a point from the center of the mask to use as a centroid...
                                ctrRC=get_center_point(depthT, combo_mask, YS.cl_boxes[which_boxes[box_idx]])

                                maskT=connected_components_filter(ctrRC,depthT, combo_mask, neighborhood=10)
                                # Combine masks from multiple objects
                                if filtered_maskT is None:
                                    filtered_maskT=maskT
                                else:
                                    filtered_maskT=filtered_maskT*maskT

                            with open(save_fName,'wb') as handle:
                                pickle.dump(filtered_maskT, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as e:
                        print("Exception " + str(e) +" - skipping")
                        continue

                    pts=torch.stack([x[filtered_maskT],y[filtered_maskT],depthT[filtered_maskT],torch.ones(((filtered_maskT>0).sum()),device=DEVICE)],dim=1)
                    pts_rot=torch.matmul(M,pts.transpose(0,1))
                    pts_rot=pts_rot[:3,:].transpose(0,1)
                    if pts_rot.shape[0]>100:
                        pclouds[cls]['xyz']=np.vstack((pclouds[cls]['xyz'],pts_rot.cpu().numpy()))
                        pclouds[cls]['rgb']=np.vstack((pclouds[cls]['rgb'],colorT[filtered_maskT].cpu().numpy()))
        except Exception as e:
            continue
    return pclouds