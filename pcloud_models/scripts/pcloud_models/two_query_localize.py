# import open3d as o3d
import os
import json
import glob
from pcloud_models.rgbd_file_list import rgbd_file_list
import pdb
import numpy as np
import json
import open3d as o3d
import pickle
from determine_relationship import get_distinct_clusters

ROOT_DIR="/home/ebeowulf/data/scannet/scans/"
IOU_THRESHOLD=0.05
MAIN_TARGET="backpack"
LLM_TARGET="strap-on bag for carrying"

def calculate_iou(minA,maxA,minB,maxB):
    deltaA=maxA-minA
    areaA=np.prod(deltaA)
    deltaB=maxB-minB
    areaB=np.prod(deltaB)
    isct_min=np.vstack((minA,minB)).max(0)
    isct_max=np.vstack((maxA,maxB)).min(0)
    isct_delta=isct_max-isct_min
    if (isct_delta<=0).sum()>0:
        return 0.0
    areaI=np.prod(isct_delta)
    return areaI / (areaA + areaB - areaI)

def build_test_data_structure(root_dir:str, main_tgt:str, llm_tgt:str):
    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(root_dir+"/*")
    test_files=dict()
    for fName in txt_files:
        fList=rgbd_file_list(fName+"/raw_output",fName+"/raw_output",fName+"/raw_output/save_results")
        aFile=fList.get_annotation_file()
        raw_main=fList.get_combined_raw_fileName(main_tgt)
        raw_llm=fList.get_combined_raw_fileName(llm_tgt)

        if os.path.exists(aFile) and os.path.exists(raw_main) and os.path.exists(raw_llm):
            with open(aFile) as fin:
                annot=json.load(fin)
            if main_tgt in annot:
                test_files[fName]={'label_file': aFile, 'raw_main': raw_main, 'raw_llm': raw_llm}
                test_files[fName]['annotation']=annot[main_tgt]
    return test_files

def build_pcloud(object_raw, initial_threshold=0.5, draw=False):
    whichP=(object_raw['probs']>=initial_threshold)
    pcd=o3d.geometry.PointCloud()
    xyzF=object_raw['xyz'][whichP]
    F2=np.where(np.isnan(xyzF).sum(1)==0)
    xyzF2=xyzF[F2]        
    pcd.points=o3d.utility.Vector3dVector(xyzF2)
    if draw:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def create_object_clusters(object_raw_file, initial_threshold=0.5):
    with open(object_raw_file,'rb') as handle:
        pcl_raw=pickle.load(handle)

    # pcd=build_pcloud(pcl_raw, draw=False)
    whichP=(pcl_raw['probs']>=initial_threshold)
    pcd=o3d.geometry.PointCloud()
    xyzF=pcl_raw['xyz'][whichP]
    F2=np.where(np.isnan(xyzF).sum(1)==0)
    xyzF2=xyzF[F2]        
    pcd.points=o3d.utility.Vector3dVector(xyzF2)
    object_clusters=get_distinct_clusters(pcd)

    probF=pcl_raw['probs'][whichP]
    probF2=probF[F2]
    for idx in range(len(object_clusters)):
        object_clusters[idx].estimate_probability(xyzF2,probF2)

    return object_clusters

def combine_matches(test_file_dict):
    try:
        objects_main=create_object_clusters(test_file_dict['raw_main'])
        objects_llm=create_object_clusters(test_file_dict['raw_llm'])
    except Exception as e:
        return None

    positive_clusters=[]
    negative_clusters=[]
    matches=np.zeros((len(objects_main),len(test_file_dict['annotation'])),dtype=int)
    for idx0 in range(len(objects_main)):
        # Match with other clusters
        cl_stats=[idx0, objects_main[idx0].prob_stats['max'], objects_main[idx0].prob_stats['mean'], -1, -1]
        for idx1 in range(len(objects_llm)):
            IOU=calculate_iou(objects_main[idx0].box[0],objects_main[idx0].box[1],objects_llm[idx1].box[0],objects_llm[idx1].box[1])
            if IOU>0.5:
                # For now - just update the prob stats on the original cloud
                cl_stats[2]=max(cl_stats[2],objects_llm[idx1].prob_stats['max'])
                cl_stats[3]=max(cl_stats[3],objects_llm[idx1].prob_stats['mean'])

        # Match with annotations
        for idxA, annot in enumerate(test_file_dict['annotation']):
            IOU_a=calculate_iou(objects_main[idx0].box[0],objects_main[idx0].box[1], np.array(annot[:3]), np.array(annot[3:]))
            if IOU_a>IOU_THRESHOLD:
                matches[idx0,idxA]=1
                positive_clusters.append(cl_stats)
            else:
                negative_clusters.append(cl_stats)
    return matches, positive_clusters, negative_clusters

if __name__ == '__main__':
    test_files=build_test_data_structure(ROOT_DIR,MAIN_TARGET, LLM_TARGET)

    save_file=ROOT_DIR + '/all_clusters.json'
    if os.path.exists(save_file):
        with open(save_file, 'r') as handle:
            all_results=json.load(handle)
    else:
        all_results=dict()
        for scene in test_files.keys():
            matches, positive_clusters, negative_clusters = combine_matches(test_files[scene])
            all_results[scene]={'matches': matches, 'positive_clusters': positive_clusters, 'negative_clusters': negative_clusters}
        pdb.set_trace()
        with open(save_file, 'w') as handle:
            json.dump(all_results, handle)
    pdb.set_trace()
