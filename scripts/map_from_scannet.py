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

def load_params(info_file):
    # Rotating the mesh to axis aligned
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=' ')

    if 'axisAlignment' not in info_dict:
        info_dict['rot_matrix'] = torch.tensor(np.identity(4))
    else:
        info_dict['rot_matrix'] = torch.tensor(info_dict['axisAlignment'].reshape(4, 4))    
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

def pointcloud_open3d(xyz_points,rgb_points=None,fileName=None, max_num_points=20000):
    pcd=o3d.geometry.PointCloud()
    if xyz_points.shape[0]<max_num_points:
        pcd.points = o3d.utility.Vector3dVector(xyz_points)
        pcd.colors = o3d.utility.Vector3dVector(rgb_points[:,[2,1,0]]/255) 
    else:
        rr=np.random.choice(np.arange(xyz_points.shape[0]),max_num_points)
        pcd.points = o3d.utility.Vector3dVector(xyz_points[rr,:])
        rgb2=rgb_points[rr,:]
        pcd.colors = o3d.utility.Vector3dVector(rgb2[:,[2,1,0]]/255) 

    o3d.visualization.draw_geometries([pcd])
    if fileName is not None:
        o3d.io.write_point_cloud(fileName,pcd)

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
        combined_xyz=np.vstack((combined_xyz,pts_rot.transpose(0,1)[:,:3]))
        combined_rgb=np.vstack((combined_rgb,colorT[depth_mask].numpy()))

        count+=1
        if count==howmany_files:
            pointcloud_open3d(fileName,combined_xyz,rgb_points=combined_rgb, write_file=True)
            return

def create_pclouds(tgt_classes:list, all_files, params, conf_threshold=0.5):
    YS=yolo_segmentation()
    height=int(params['colorHeight'])
    width=int(params['colorWidth'])

    rows=torch.tensor(np.tile(np.arange(height).reshape(height,1),(1,width))-params['my_color'])
    cols=torch.tensor(np.tile(np.arange(width),(height,1))-params['mx_color'])

    pclouds=dict()
    for cls in tgt_classes:
        pclouds[cls]={'xyz': np.zeros((0,3),dtype=float),'rgb': np.zeros((0,3),dtype=np.uint8)}

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
            depthT=torch.tensor(depthI.astype('float')/1000.0)
            colorT=torch.tensor(colorI)
            x = cols*depthT/params['fx_color']
            y = rows*depthT/params['fy_color']
            depth_mask=(depthT>1e-4)*(depthT<10.0)

            # Build the rotation matrix
            M=torch.matmul(params['rot_matrix'],torch.tensor(all_files[key]['pose']))

            # Now extract a mask per category
            for cls in tgt_classes:
                cls_mask=YS.get_mask(cls)
                if cls_mask is not None:
                    combo_mask=(torch.tensor(cls_mask)>conf_threshold)*depth_mask
                    pts=torch.stack([x[combo_mask],y[combo_mask],depthT[combo_mask],torch.ones(((combo_mask>0).sum()))],dim=1)
                    pts_rot=torch.matmul(M,pts.transpose(0,1))
                    pclouds[cls]['xyz']=np.vstack((pclouds[cls]['xyz'],pts_rot[:3,:].transpose(0,1)))
                    pclouds[cls]['rgb']=np.vstack((pclouds[cls]['rgb'],colorT[combo_mask].numpy()))
        except Exception as e:
            continue
    pdb.set_trace()
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
    params=load_params(args.param_file)
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
        pointcloud_open3d(pclouds[key]['xyz'],pclouds[key]['rgb'],fileName=args.scan_directory+"/"+key+".ply")

    pdb.set_trace()

