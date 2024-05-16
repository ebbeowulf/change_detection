from yolo_segmentation import yolo_segmentation
import cv2
import numpy as np
import argparse
import glob
import pdb
import pickle
import os
import torch
import open3d as o3d

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def load_camera_info(info_file):
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            if key=='sceneType':
                if val[-1]=='\n':
                    val=val[:-1]
                info_dict[key] = val
            elif key=='axisAlignment':
                info_dict[key] = np.fromstring(val, sep=' ')
            else:
                info_dict[key] = float(val)

    if 'axisAlignment' not in info_dict:
       info_dict['rot_matrix'] = np.identity(4)
    else:
        info_dict['rot_matrix'] = info_dict['axisAlignment'].reshape(4, 4)
    return info_dict

def read_scannet_pose(pose_fName):
    # Get the pose - 
    try:
        with open(pose_fName,'r') as fin:
            LNs=fin.readlines()
            pose=np.zeros((4,4),dtype=float)
            for r_idx,ln in enumerate(LNs):
                if ln[-1]=='\n':
                    ln=ln[:-1]
                p_split=ln.split(' ')
                for c_idx, val in enumerate(p_split):
                    pose[r_idx, c_idx]=float(val)
        return pose
    except Exception as e:
        return None
    
def build_file_structure(input_dir):
    all_files=dict()
    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(input_dir+'/*.txt')
    for fName in txt_files:
        try:
            ppts=fName.split('.')
            rootName=ppts[0]
            number=int(ppts[0].split('-')[-1])
            all_files[number]={'root': rootName, 'poseFile': fName, 'depthFile': rootName+'.depth_reg.png', 'colorFile': rootName+'.color.jpg'}
        except Exception as e:
            continue
    return all_files

def load_all_poses(all_files:dict):
    for key in all_files.keys():
        pose=read_scannet_pose(all_files[key]['poseFile'])
        all_files[key]['pose']=pose

def process_images(all_files:dict):
    print("process_images")
    YS=yolo_segmentation()
    for key in all_files.keys():
        print(all_files[key]['colorFile'])
        pkl=all_files[key]['colorFile']+".pkl"
        if not os.path.exists(pkl):
            img,results=YS.process_file(all_files[key]['colorFile'])
            with open(pkl, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        all_files[key]['yolo']=pkl

def create_object_list(all_files):
    YS=yolo_segmentation()
    obj_list=dict()
    for key in all_files.keys():
        with open(all_files[key]['yolo'], 'rb') as handle:
            results=pickle.load(handle)
            YS.load_prior_results(results)
        for id, prob in zip(YS.cl_labelID, YS.cl_probs):
            if YS.id2label[id] not in obj_list:
                obj_list[YS.id2label[id]]={'images': [key], 'probs': [prob]}
            else:
                obj_list[YS.id2label[id]]['images'].append(key)
                obj_list[YS.id2label[id]]['probs'].append(prob)
    return obj_list

def get_high_confidence_objects(obj_list, confidence_threshold=0.5):
    o_list=[]
    for key in obj_list:
        maxV=max(obj_list[key]['probs'])
        if maxV>=confidence_threshold:
            o_list.append(key)
    return o_list

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

def visualize_combined_xyzrgb(fileName, all_files, params, howmany_files=100):
    height=int(params['colorHeight'])
    width=int(params['colorWidth'])

    rows=torch.tensor(np.tile(np.arange(height).reshape(height,1),(1,width))-params['my_color'])
    cols=torch.tensor(np.tile(np.arange(width),(height,1))-params['mx_color'])

    combined_xyz=np.zeros((0,3),dtype=float)
    combined_rgb=np.zeros((0,3),dtype=np.uint8)

    count=0
    for key in range(max(all_files.keys())):
        if key not in all_files:
            continue

        # Create the generic depth data
        colorI=cv2.imread(all_files[key]['colorFile'], -1)
        depthI=cv2.imread(all_files[key]['depthFile'], -1)
        depthT=torch.tensor(depthI.astype('float')/1000.0)
        colorT=torch.tensor(colorI)
        x = cols*depthT/params['fx_color']
        y = rows*depthT/params['fy_color']
        depth_mask=(depthT>1e-4)*(depthT<10.0)

        # Rotate the points into the right space
        M=torch.matmul(params['rot_matrix'],torch.tensor(all_files[key]['pose']))
        pts=torch.stack([x[depth_mask],y[depth_mask],depthT[depth_mask],torch.ones(((depth_mask>0).sum()))],dim=1)
        pts_rot=torch.matmul(M,pts.transpose(0,1))

        count+=1
        if count==howmany_files:
            break
    pointcloud_open3d(fileName,combined_xyz,rgb_points=combined_rgb, write_file=True)
    return

# def compare_points(cl1,cl2,max_dist):
#     if len(cl1.shape)==1 or cl1.shape[0] == 1:
#         X1=cl1.reshape((1,3))
#     else:
#         X1=cl1
#     if len(cl2.shape)==1 or cl2.shape[0] == 1:
#         X2=cl2.reshape((1,3))
#     else:
#         X2=cl2

#     dists=np.zeros((X1.shape[0],X2.shape[0]),dtype=float)
#     for i in range(X2.shape[0]):
#         dists[i, :] = np.sqrt(np.sum((X2[i] - X1) ** 2, axis=1))
#     return dists.min()<max_dist
    

# def compare_clusters(cl1, cl2, max_dist):
#     return compare_points(cl1['pts'],cl2['pts'],max_dist)

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

def connected_components_filter(centerRC, depthT:torch.tensor, maskI:torch.tensor, neighborhood=4, max_depth_dist=0.1):
    queue=[centerRC]
    # height=depthT.shape[0]
    # width=depthT.shape[1]
    cc_mask=torch.zeros(maskI.shape,dtype=torch.uint8,device=DEVICE)
    cc_mask[centerRC[0],centerRC[1]]=2
    # rows=torch.tensor(np.tile(np.arange(height).reshape(height,1),(1,width)),device=DEVICE)
    # cols=torch.tensor(np.tile(np.arange(width),(height,1)),device=DEVICE)    
    iterations=0
    # sampleMatrix=torch.ones((2*neighborhood,2*neighborhood),dtype=bool,device=DEVICE)
    # sampleMatrix[2:-2,2:-2]=False
    while len(queue)>0:
        point=queue.pop(0)
        target_depth = depthT[point[0],point[1]]
        
        minR=max(0,point[0]-neighborhood)
        minC=max(0,point[1]-neighborhood)
        maxR=min(depthT.shape[0],point[0]+neighborhood)
        maxC=min(depthT.shape[1],point[1]+neighborhood)
        regionD=depthT[minR:maxR,minC:maxC]
        regionMask=maskI[minR:maxR,minC:maxC]
        # rowMask=rows[minR:maxR,minC:maxC]
        # colMask=cols[minR:maxR,minC:maxC]
        regionCC=cc_mask[minR:maxR,minC:maxC]

        reachableAreaMask=((regionD-target_depth).abs()<max_depth_dist)*regionMask
        localMask=reachableAreaMask*(regionCC==0)
        # regionCC=localMask+1

        # Update the queue
        if 1: # randomsample
            sample_size=5
            indices=localMask.nonzero()+torch.tensor([minR,minC],device=DEVICE)
            if len(indices)<sample_size:
                queue=queue+indices.tolist()
            else:
                rr=np.random.choice(np.arange(len(indices)),sample_size)
                queue=queue+indices[rr].tolist()
        # elif 1: # use sample matrix of edges only
        #     FullQ=torch.vstack((rowMask[localMask*sampleMatrix],colMask[localMask*sampleMatrix])).transpose(0,1)
        #     queue=queue+FullQ.cpu().tolist()
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
    # cv2.imshow("mask",cc_mask.cpu().numpy()*100)
    # cv2.waitKey(1)
    return cc_mask==2
    # pdb.set_trace()

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

def create_pclouds(tgt_classes:list, all_files, params, conf_threshold=0.5):
    YS=yolo_segmentation()
    height=int(params['colorHeight'])
    width=int(params['colorWidth'])

    rows=torch.tensor(np.tile(np.arange(height).reshape(height,1),(1,width))-params['my_color'],device=DEVICE)
    cols=torch.tensor(np.tile(np.arange(width),(height,1))-params['mx_color'],device=DEVICE)

    pclouds=dict()
    for cls in tgt_classes:
        pclouds[cls]={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8)}

    rot_matrixT=torch.tensor(params['rot_matrix'],device=DEVICE)
    for key in range(max(all_files.keys())):
        if key not in all_files:
            continue
        print(key)
        try:
            with open(all_files[key]['yolo'], 'rb') as handle:
                results=pickle.load(handle)
                YS.load_prior_results(results)
            colorI=cv2.imread(all_files[key]['colorFile'], -1)
            depthI=cv2.imread(all_files[key]['depthFile'], -1)
            depthT=torch.tensor(depthI.astype('float')/1000.0,device=DEVICE)
            colorT=torch.tensor(colorI,device=DEVICE)
            x = cols*depthT/params['fx_color']
            y = rows*depthT/params['fy_color']
            depth_mask=(depthT>1e-4)*(depthT<10.0)

            # Build the rotation matrix
            M=torch.matmul(rot_matrixT,torch.tensor(all_files[key]['pose'],device=DEVICE))

            # Now extract a mask per category
            for cls in tgt_classes:
                cls_mask=YS.get_mask(cls)
                if cls_mask is not None:
                    save_fName=all_files[key]['root']+"."+cls+".pkl"
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_directory',type=str,help='location of raw images to process')
    parser.add_argument('param_file',type=str,help='camera parameter file for this scene')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    # parser.add_argument('--targets',type=list, nargs='+', default=None, help='Set of target classes to build point clouds for')
    args = parser.parse_args()
    all_files=build_file_structure(args.scan_directory)
    params=load_camera_info(args.param_file)
    load_all_poses(all_files)
    process_images(all_files)
    # create_map(args.tgt_class,all_files)
    # obj_list=create_object_list(all_files)
    # obj_list=['bed','vase','potted plant','tv','refrigerator','chair']
    # create_combined_xyzrgb(obj_list, all_files, params)
    if args.targets is None:
        obj_list=create_object_list(all_files)
        print("Detected Objects (Conf > 0.75)")
        high_conf_list=get_high_confidence_objects(obj_list, confidence_threshold=0.75)
        print(high_conf_list)
        pclouds=create_pclouds(high_conf_list,all_files,params,conf_threshold=0.5)
    else:
        pclouds=create_pclouds(args.targets,all_files,params,conf_threshold=0.5)
    for key in pclouds.keys():
        fileName=args.scan_directory+"/"+key+".ply"
        pcd=pointcloud_open3d(pclouds[key]['xyz'],pclouds[key]['rgb'])
        if 0:
            o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(fileName,pcd)

