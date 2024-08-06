import open3d as o3d
import os
import json
import glob
from pcloud_models.rgbd_file_list import rgbd_file_list
import pdb
import numpy as np
from sklearn.cluster import DBSCAN
import pickle

ROOT_DIR="/home/emartinso/data/scannet/scans/"
TGT_CLASS='backpack'
IOU_THRESHOLD=0.1

def build_test_data_structure(root_dir:str, tgt_object:str):
    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(root_dir+"/*")
    test_files=dict()
    for fName in txt_files:
        fList=rgbd_file_list(fName+"/raw_output",fName+"/raw_output",fName+"/raw_output/save_results")
        aFile=fList.get_annotation_file()
        plyFile=fList.get_combined_pcloud_fileName(tgt_object)
        rawFile=fList.get_combined_raw_fileName(tgt_object)
        if os.path.exists(aFile) and os.path.exists(plyFile) and os.path.exists(rawFile):
            with open(aFile) as fin:
                annot=json.load(fin)
            if tgt_object in annot:
                test_files[fName]={'label_file': aFile, 'ply_file': plyFile, 'raw_ply_file': rawFile}
                test_files[fName]['annotation']=annot[tgt_object]
            # else:
            #     test_files[fName]['annotation']=annot[tgt_object]
    return test_files

def build_clusters(xyz, prob, threshold:float, grid_cell_size=0.01, eps=0.02, min_count=[10]):
    pcd=o3d.geometry.PointCloud()
    F1=np.where(prob>threshold)
    xyzF=xyz[F1]
    F2=np.where(np.isnan(xyzF).sum(1)==0)
    pcd.points=o3d.utility.Vector3dVector(xyzF[F2])
    pcd_small=pcd.voxel_down_sample(grid_cell_size)
    clusters=dict()
    for cnt in min_count:
        if F1[0].shape[0]>0:
            clusters[cnt]=DBSCAN(eps=eps, min_samples=cnt).fit(np.array(pcd_small.points))
        else:
            clusters[cnt]=None
    return clusters, pcd_small

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

# Return the following:
#      number of matched annotations
#      number of matched clusters
#      number of unmatched clusters
#      number of unmatched annotations
def compare_annotations(annotations, clusters, pcloud):
    if clusters is None or clusters.labels_.max()<0:
        return [0,0,0,len(annotations)]

    max_cl_id=clusters.labels_.max()

    if len(annotations)<1:
        return [0,0,max_cl_id+1,0]

    matchA=np.zeros((len(annotations)),dtype=bool)
    matchC=np.zeros((max_cl_id+1),dtype=bool)
    all_pts=np.array(pcloud.points)
    for cl_idx in range(max_cl_id+1):
        try:
            whichP=np.where(clusters.labels_==cl_idx)
            cl_array=all_pts[whichP]
            box_min=cl_array.min(0)
            box_max=cl_array.max(0)
            for a_idx, annot in enumerate(annotations):
                iou=calculate_iou(box_min, box_max, np.array(annot[:3]),np.array(annot[3:]))
                if iou>IOU_THRESHOLD:
                    matchA[a_idx]=True
                    matchC[cl_idx]=True
                    break
        except Exception as e:
            print(f"Exception: {e}")
            pdb.set_trace()
    
    return [matchC.sum(),matchA.sum(),(matchC==False).sum(),(matchA==False).sum()]

def eval_ply(test_files:dict, save_file:str):
    try:
        with open(save_file,'r') as fin:
            results=json.load(fin)
    except Exception as e:
        results=dict()

    for key in test_files:
        # Skip those files that have already been evaluated and saved to the save_file
        if key in results:
            continue
        # key='/home/emartinso/data/scannet/scans/scene0065_01'
        print(key)
        results[key]={'stats': [], 'matches': []}
        try:
            with open(test_files[key]['raw_ply_file'], 'rb') as handle:
                raw_pts=pickle.load(handle)
        except Exception as e:
            print(f"Error loading ply file: {e}")
            pdb.set_trace()

        cnts=np.arange(500,3100,500).tolist()
        for threshold in np.arange(0.75,1.0,0.03):
            clusters,pcd_small=build_clusters(raw_pts['xyz'],raw_pts['probs'],threshold,min_count=cnts)
            for cl_cnt in clusters:                
                matches=compare_annotations(test_files[key]['annotation'],clusters[cl_cnt],pcd_small)
                results[key]['stats'].append([threshold,cl_cnt])
                results[key]['matches'].append(matches)
        
        # Convert numpy types to serializable types
        results[key]['stats']=np.array(results[key]['stats']).astype(float).tolist()
        results[key]['matches']=np.array(results[key]['matches']).astype(int).tolist()
        with open(save_file,'w') as fout:
            json.dump(results,fout)
    return results

test_files=build_test_data_structure(ROOT_DIR,TGT_CLASS)
results=eval_ply(test_files,TGT_CLASS+".json")
# load_annotations(test_files,TGT_CLASS)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('image',type=str,help='location of image to process')
#     parser.add_argument('tgt_prompt',type=str,default=None,help='specific prompt for clip class')
#     parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
#     args = parser.parse_args()

#     CS=clip_seg_depth([args.tgt_prompt])
#     image=CS.process_file(args.image, threshold=args.threshold)
#     mask=CS.get_mask(0)
#     if mask is None:
#         print("Something went wrong - no mask to display")
#     else:
#         image=image.astype(float)/255.0
#         IM=cv2.bitwise_and(image[:,:,0],image[:,:,0],mask=mask.astype(np.uint8))
#         cv2.imshow("res",IM)
#         cv2.waitKey(0)
    
