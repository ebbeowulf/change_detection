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

#ROOT_DIR="/home/ebeowulf/data/scannet/scans/"
ROOT_DIR="/data3/datasets/scannet/scans/"
IOU_THRESHOLD=0.05
FLOOR_THRESHOLD=0.05

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
        else:
            print(f"build_test_data_structure> disarding {fName}...")
    pdb.set_trace()
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

def create_object_clusters(object_raw_file, floor_threshold=0.1, detection_threshold=0.5):
    with open(object_raw_file,'rb') as handle:
        pcl_raw=pickle.load(handle)

    # pcd=build_pcloud(pcl_raw, draw=False)
    if pcl_raw['xyz'].shape[0]==0:
        return []
    whichP=(pcl_raw['probs']>=detection_threshold)
    pcd=o3d.geometry.PointCloud()
    xyzF=pcl_raw['xyz'][whichP]
    F2=np.where(np.isnan(xyzF).sum(1)==0)
    xyzF2=xyzF[F2]        
    pcd.points=o3d.utility.Vector3dVector(xyzF2)
    object_clusters=get_distinct_clusters(pcd, floor_threshold=floor_threshold)

    probF=pcl_raw['probs'][whichP]
    probF2=probF[F2]
    for idx in range(len(object_clusters)):
        object_clusters[idx].estimate_probability(xyzF2,probF2)

    return object_clusters

def combine_matches(test_file_dict, detection_threshold=0.5):
    try:
        objects_main=create_object_clusters(test_file_dict['raw_main'], FLOOR_THRESHOLD, detection_threshold)
        objects_llm=create_object_clusters(test_file_dict['raw_llm'], FLOOR_THRESHOLD, detection_threshold)
    except Exception as e:
        print(f"Exception: {e}")
        pdb.set_trace()
        return [], [] ,[]

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
                cl_stats[3]=max(cl_stats[3],objects_llm[idx1].prob_stats['max'])
                cl_stats[4]=max(cl_stats[4],objects_llm[idx1].prob_stats['mean'])

        # Match with annotations        
        for idxA, annot in enumerate(test_file_dict['annotation']):
            IOU_a=calculate_iou(objects_main[idx0].box[0],objects_main[idx0].box[1], np.array(annot[:3]), np.array(annot[3:]))
            if IOU_a>IOU_THRESHOLD:
                matches[idx0,idxA]=1
        if matches[idx0,:].sum()>0:
            positive_clusters.append(cl_stats)
        else:
            negative_clusters.append(cl_stats)
    return matches, positive_clusters, negative_clusters

def estimate_likelihood(cluster_stats, category):
    if category<4:
        return cluster_stats[category+1]
    elif category==4: # combined max
        return cluster_stats[1]*cluster_stats[3]
        # return (cluster_stats[1]+cluster_stats[3])/2
    elif category==5: # combined max
        return cluster_stats[2]*cluster_stats[4]
        # return (cluster_stats[2]+cluster_stats[4])/2
    else:
        raise(Exception(f"Category {category} not implemented yet"))
    
# def build_cluster_struct(all_results, category):
#     positive=[]
#     negative=[]
#     for scene in all_results:
#         for pos in all_results[scene]['positive_clusters']:
#             positive.append(estimate_likelihood(pos, category))
#         for neg in all_results[scene]['negative_clusters']:
#             negative.append(estimate_likelihood(neg, category))
#     return np.array(positive), np.array(negative)

def trapz(y,x):
    ym=np.mean(np.vstack((y[:-1],y[1:])),0)
    width=np.abs(np.diff(x))
    return (width*ym).sum()

def get_graph_stats_per_scene(matches, positive_cl, negative_cl, cat_, threshold):
    # TP = number of object blobs successfully matched
    # FN = number of annotations not matched to a blob
    # FP = number of clusters that do not correspond to a real object
    # TN = number of clusters discarded by combination of category + threshold
    stats=np.zeros((4),dtype=float) #TP/FN/FP/TN
    if matches.shape[0]==0:
        if matches.shape[1]>0:
            stats[1]+=matches.shape[1]
        return stats
    # Start with re-estimating probability for each cluster
    all_blob_prob=np.zeros((matches.shape[0]),dtype=float)-1
    prob=np.zeros((matches.shape[1]),dtype=float)-1
    for pos_ in positive_cl:
        lk=estimate_likelihood(pos_, cat_)
        all_blob_prob[pos_[0]]=max(lk,all_blob_prob[pos_[0]])
    for neg_ in negative_cl:
        lk=estimate_likelihood(neg_, cat_)
        all_blob_prob[neg_[0]]=max(lk,all_blob_prob[neg_[0]])
    
    positive_clusters=(all_blob_prob>=threshold)
    count_positive_clusters=positive_clusters.sum()
    
    if matches.shape[1]<1:
        matched_positive_clusters=0
        unmatched_annotations=0
        count_true_negative_clusters=all_blob_prob.shape[0]-count_positive_clusters    
    else:
        whichP=(matches.sum(1)>0)
        matched_positive_clusters=(positive_clusters*whichP).sum()
        unmatched_annotations=(matches.sum(0)<1).sum()
        count_true_negative_clusters=((~positive_clusters)*(~whichP)).sum()

    stats[0]+=matched_positive_clusters #TP
    stats[1]+=unmatched_annotations     #FN
    stats[2]+=count_positive_clusters - matched_positive_clusters #FP
    stats[3]+=count_true_negative_clusters

    return stats        
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('main_target', type=str, help='main target search string')
    parser.add_argument('llm_target', type=str, help='llm target search string')
    parser.add_argument('--threshold_low',type=str,default=0.5, help='low end of the threshold range')
    parser.add_argument('--threshold_high',type=str,default=1.0, help='high end of the threshold range')
    parser.add_argument('--threshold_delta',type=str,default=0.03, help='delta of the threshold range')
    parser.add_argument('--save_dir',type=str,default=None, help='where are the intermediate files saved? By default these are saved to the root dir')
    args = parser.parse_args()

    test_files=build_test_data_structure(ROOT_DIR,args.main_target, args.llm_target)
    thresholds=np.arange(args.threshold_low, args.threshold_high, args.threshold_delta)

    all_legends=['max', 'mean', 'max-llm', 'mean-llm', 'max-combo', 'mean-combo']
    # cat_list=[0,1,2,3,4,5]
    cat_list=[1,3,5]
    legend=[ all_legends[cat_] for cat_ in cat_list ]

    recallN = np.zeros((thresholds.shape[0],len(cat_list)))
    precisionN = np.zeros((thresholds.shape[0],len(cat_list)))
    specificityN = np.zeros((thresholds.shape[0],len(cat_list)))
    AUC=np.zeros(len(cat_list))

    for t_idx in range(len(thresholds)):
        threshold=thresholds[t_idx]
        if args.save_dir is None:
            save_file=ROOT_DIR + f"/{args.main_target}_all_clusters{threshold:.2f}.pkl"
        else:
            save_file=args.save_dir + f"/{args.main_target}_all_clusters{threshold:.2f}.pkl"
        if os.path.exists(save_file):
            with open(save_file, 'rb') as handle:
                all_results=pickle.load(handle)
        else:
            all_results=dict()
            for scene in test_files.keys():
                print(scene)
                matches, positive_clusters, negative_clusters = combine_matches(test_files[scene],detection_threshold=threshold)
                print(matches)
                all_results[scene]={'matches': matches, 'positive_clusters': positive_clusters, 'negative_clusters': negative_clusters}
            with open(save_file, 'wb') as handle:
                pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for cat_idx, cat_ in enumerate(cat_list):
            stats=np.zeros((4),dtype=float)
            for scene in all_results:
                stats=stats + get_graph_stats_per_scene(all_results[scene]['matches'], all_results[scene]['positive_clusters'], all_results[scene]['negative_clusters'], cat_, threshold)
            TP=stats[0]
            FN=stats[1]
            FP=stats[2]
            TN=stats[3]
            precisionN[t_idx, cat_idx]=TP/(TP+FP+1e-6)
            recallN[t_idx, cat_idx]=TP/(TP+FN+1e-6)
            specificityN[t_idx, cat_idx]=TN/(FP+TN+1e-6)
            
    pdb.set_trace()
            
    # pdb.set_trace()
    import matplotlib.pyplot as plt
    for cat_idx, cat_ in enumerate(cat_list):
        plt.figure(1)
        plt.plot(recallN[:,cat_idx], precisionN[:, cat_idx])
        # plt.figure(2)
        # plt.plot(1-specificityN[:,cat_idx], recallN[:,cat_idx])
    plt.figure(1)
    plt.legend(legend)
    plt.title('Recall vs Precision')
    # plt.figure(2)
    # plt.legend(legend)
    # plt.title('ROC')
    plt.show()

        

