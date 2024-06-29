from map_from_scannet import load_camera_info, read_scannet_pose
from camera_params import camera_params
from rgbd_file_list import rgbd_file_list
import argparse
import pdb
import os
import glob
import cv2
import numpy as np

#LABEL_CSV="/data2/datasets/scannet/scannetv2-labels.combined.tsv"
LABEL_CSV="/home/emartinso/data/scannet/scannetv2-labels.combined.tsv"
ID2STR=None
STR2ID=None
def read_label_csv():
    global ID2STR, STR2ID
    with open(LABEL_CSV,"r") as fin:
        lines=fin.readlines()
    categories=lines[0].split('\t')
    id_idx=categories.index('id')
    cat_idx=categories.index('category')
    ID2STR=dict()
    STR2ID=dict()    
    for line in lines[1:]:
        lineS=line.split('\t')
        id=int(lineS[id_idx])
        if id in ID2STR:
            if lineS[cat_idx] not in ID2STR[id]:
                ID2STR[id].append(lineS[cat_idx])
        else:
            ID2STR[id]=[lineS[cat_idx]]
        if lineS[cat_idx] in STR2ID:
            if id not in STR2ID[lineS[cat_idx]]:
                STR2ID[lineS[cat_idx]].append(id)
        else:
            STR2ID[lineS[cat_idx]]=[id]

def id_from_string(obj_name):
    global STR2ID
    if STR2ID is None:
        read_label_csv()
    if obj_name in STR2ID:
        return STR2ID[obj_name]
    return None

def string_from_id(id):
    global ID2STR
    if ID2STR is None:
        read_label_csv()
    if id in ID2STR:
        return ID2STR[id]
    return None

def build_file_structure(raw_dir, label_dir, save_dir):
    fList = rgbd_file_list(label_dir, raw_dir, save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(raw_dir+'/*.txt')
    for fName in txt_files:
        try:
            ppts=fName.split('.')
            rootName=ppts[0].split('/')[-1]
            number=int(rootName.split('-')[-1])
            pose=read_scannet_pose(fName)
            fList.add_file(number,"%d.png"%(number),rootName+'.depth_reg.png')
            fList.add_pose(number, pose)
        except Exception as e:
            continue
    return fList

def identify_all_labels(fList:rgbd_file_list):
    import json
    all_labels_file=fList.intermediate_save_dir+"/all_labels.json"
    if os.path.exists(all_labels_file):
        try:
            with open(all_labels_file, 'r') as handle:
                ldict=json.load(handle)
            # Need to convert keys to ints
            label_dict={int(key):ldict[key] for key in ldict}
            return label_dict
        except Exception as e:
            print("Exception loading labels dict: " + str(e))

    label_dict=dict()
    for key in range(max(fList.keys())):
        if not fList.is_key(key):
            continue
        # Create the generic depth data
        print(fList.get_color_fileName(key))
        labelI=cv2.imread(fList.get_color_fileName(key), -1)
        lbls=np.unique(labelI)
        for idx in range(lbls.shape[0]):
            lbl_id=int(lbls[idx])
            if lbl_id not in label_dict:
                label_dict[lbl_id]=[]
            label_dict[lbl_id].append(key)

    with open(all_labels_file, 'w') as handle:
        json.dump(label_dict, handle)

    return label_dict

# Combine together multiple point clouds into a single
#   cloud and display the result using open3d. 
def visualize_combined_label(fList:rgbd_file_list, params:camera_params, object_or_id:str, max_num_points=2000000):
    label_dict = identify_all_labels(fList)
    if object_or_id.isnumeric():
        label_ids=[int(object_or_id)]
    else:
        label_ids = id_from_string(object_or_id)
    
    print(f"Labels found include: {[string_from_id(val) for val in list(label_dict.keys())]}")


    all_keys=[]
    for id in label_ids:
        if id in label_dict:
            all_keys=all_keys+label_dict[id]
    if len(all_keys)<1:
        print("Label not found in these images - exiting")
        return None

    all_keys=np.unique(all_keys).tolist()

    #Prepare for poincloud creation
    import torch
    from map_utils import pointcloud_open3d
    rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy)
    cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx)
    combined_xyz=np.zeros((0,3),dtype=float)
    rot_matrixT=torch.tensor(params.rot_matrix)
    
    howmany=0
    for key in all_keys:
        try:
            # Create the generic depth data
            print(fList.get_color_fileName(key))
            labelI=cv2.imread(fList.get_color_fileName(key), -1)
            label_mask=(labelI==label_ids[0])
            for lbl_id in label_ids[1:]:
                label_mask=label_mask+(labelI==lbl_id)

            if label_mask.sum()==0:
                continue
            depthI=cv2.imread(fList.get_depth_fileName(key), -1)
            depthT=torch.tensor(depthI.astype('float')/1000.0)
            x = cols*depthT/params.fx
            y = rows*depthT/params.fy
            depth_mask=(depthT>1e-4)*(depthT<10.0)
            combo_mask=torch.tensor(label_mask)*depth_mask

            # Rotate the points into the right space
            M=torch.matmul(rot_matrixT,torch.tensor(fList.get_pose(key)))
            pts=torch.stack([x[combo_mask],y[combo_mask],depthT[combo_mask],torch.ones(((combo_mask>0).sum()))],dim=1)
            pts_rot=torch.matmul(M,pts.transpose(0,1))
            pts_rot=pts_rot[:3,:].transpose(0,1)

            if pts_rot.shape[0]>100:
                combined_xyz=np.vstack((combined_xyz,pts_rot.cpu().numpy()))
        except Exception as e:
            print(f"Exception loading file {key}: {e}")
    return pointcloud_open3d(combined_xyz, max_num_points=max_num_points)


if __name__ == '__main__':
    import datetime
    start=datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('object_type', type=str, help='object type to search for (see scannet/scannetv2-labels.combined.tsv for list)')
    parser.add_argument('--param_file',type=str,default=None,help='camera parameter file for this scene - default is of form <raw_dir>/scene????_??.txt')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color, depth, and pose information')
    parser.add_argument('--label_dir',type=str,default='label-filt', help='subdirectory containing the label images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--draw', dest='draw', action='store_true')
    parser.set_defaults(draw=False)    
    args = parser.parse_args()
    T1=datetime.datetime.now()


    if args.param_file is not None:
        par_file=args.param_file
    else:
        s_root=args.root_dir.split('/')
        if s_root[-1]=='':
            par_file=args.root_dir+"%s.txt"%(s_root[-2])
        else:
            par_file=args.root_dir+"/%s.txt"%(s_root[-1])
    params=load_camera_info(par_file)
    T2=datetime.datetime.now()

    fList=build_file_structure(args.root_dir+"/"+args.raw_dir, args.root_dir+"/"+args.label_dir, args.root_dir+"/"+args.save_dir)
    pcd=visualize_combined_label(fList, params, args.object_type)
    T3=datetime.datetime.now()
    if pcd is not None:
        import open3d as o3d
        labeled_file=fList.get_labeled_pcloud_fileName(args.object_type)
        o3d.io.write_point_cloud(labeled_file,pcd)
        if args.draw:
            o3d.visualization.draw_geometries([pcd])
    
    delta1=(T1-start).total_seconds()
    delta2=(T2-T1).total_seconds()
    delta3=(T3-T2).total_seconds()
    print(f"Seconds: 1){delta1}, 2){delta2}, 3){delta3}")