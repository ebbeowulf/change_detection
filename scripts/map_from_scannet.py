from yolo_segmentation import yolo_segmentation
import cv2
import numpy as np
import argparse
import glob
import pdb
import pickle
import os
import torch

def load_params(info_file):
    # Rotating the mesh to axis aligned
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=' ')

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
            all_files[number]={'root': rootName, 'poseFile': fName, 'depthFile': rootName+'.depth.pgm', 'colorFile': rootName+'.color.jpg'}
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

def create_pclouds(tgt_classes:list, all_files, params, conf_threshold):
    YS=yolo_segmentation()
    height=int(params['depthHeight'])
    width=int(params['depthWidth'])

    rows=torch.tensor(np.tile(np.arange(height).reshape(height,1),(1,width))-params['my_depth'])
    cols=torch.tensor(np.tile(np.arange(width),(height,1))-params['mx_depth'])

    pclouds=dict()
    for cls in tgt_classes:
        pclouds[cls]={'arr': np.zeros((0,3),dtype=float),'prob': np.zeros((0,1),dtype=float)}

    for key in all_files.keys():
        with open(all_files[key]['yolo'], 'rb') as handle:
            results=pickle.load(handle)
            YS.load_prior_results(results)
        
        # Create the generic depth data
        depthI=cv2.imread(all_files[key]['depthFile'], -1)
        depthT=torch.tensor(depthI.astype('float')/1000.0)
        x = cols*depthT/params['fx_depth']
        y = rows*depthT/params['fy_depth']
        depth_mask=(depthT>1e-4)*(depthT<10.0)

        # Now extract a mask per category
        
        pts=torch.stack([x[depth_mask],y[depth_mask],depthT[depth_mask],torch.zeros(((depth_mask>0).sum()))],dim=1)
        pts=torch.stack([pts,torch.zeros((pts.shape[0]),dtype=float)])

        pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scan_directory',type=str,help='location of raw images to process')
    parser.add_argument('param_file',type=str,help='camera parameter file for this scene')
    parser.add_argument('--tgt-class',type=str,default=None,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.25,help='threshold to apply during computation')
    args = parser.parse_args()
    all_files=build_file_structure(args.scan_directory)
    params=load_params(args.param_file)
    load_all_poses(all_files)
    process_images(all_files)
    # create_map(args.tgt_class,all_files)
    # obj_list=create_object_list(all_files)
    obj_list=['bed','vase','potted plant','tv','refrigerator','chair']
    create_pclouds(obj_list, all_files, params, 0.5)
    pdb.set_trace()

