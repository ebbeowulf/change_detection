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
from map_utils import get_distinct_clusters, object_pcloud
from scannet_processing import get_scene_type
import sys

#ROOT_DIR="/home/ebeowulf/data/scannet/scans/"
# ROOT_DIR="/data3/datasets/scannet/scans/"
IOU_THRESHOLD=0.05
FLOOR_THRESHOLD=-0.1
ABSOLUTE_MIN_CLUSTER_SIZE=100
DEBUG_=False

FIXED_ROOM_STATS_FILE="/data3/datasets/scannet/scans/llm_likelihoods.json"

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

def build_test_data_structure(root_dir:str, main_tgt:str, llm_tgt:str, is_pose_filtered=False, classifier_type=None):
    with open(FIXED_ROOM_STATS_FILE, 'r') as fin:
        all_room_probs=json.load(fin)
    if main_tgt not in all_room_probs:
        print(f"No room probabilities found for {main_tgt}")
        sys.exit(-1)
    room_prob=all_room_probs[main_tgt]
    max_room_prob=max([room_prob[room] for room in room_prob])

    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(root_dir+"/scene0[0,1]*")
    test_files=dict()
    for fName in txt_files:
        #We also want the scene type from params file
        s_root=fName.split('/')
        if s_root[-1]=='':
            par_file=fName+"%s.txt"%(s_root[-2])
        else:
            par_file=fName+"/%s.txt"%(s_root[-1])        
        sceneType=get_scene_type(par_file)


        fList=rgbd_file_list(fName+"/raw_output",fName+"/raw_output",fName+"/raw_output/save_results", is_pose_filtered)
        aFile=fList.get_combo_annotation_file() # use this one that merges the annotations from both directories on the server
        raw_main=fList.get_combined_raw_fileName(main_tgt, classifier_type=classifier_type)
        raw_llm=fList.get_combined_raw_fileName(llm_tgt, classifier_type=classifier_type)

        if not os.path.exists(aFile):
            print(f"build_test_data_structure> disarding {fName}... missing {aFile}")
        elif not os.path.exists(raw_main):
            print(f"build_test_data_structure> disarding {fName}... missing {raw_main}")
        elif not os.path.exists(raw_llm):
            print(f"build_test_data_structure> disarding {fName}... missing {raw_llm}")
        else:
            with open(aFile) as fin:
                annot=json.load(fin)
            if main_tgt in annot:
                if sceneType in room_prob:
                    raw_scene_prob=room_prob[sceneType]
                    normalized_scene_prob=room_prob[sceneType]/max_room_prob
                else:
                    raw_scene_prob=1.0
                    normalized_scene_prob=1.0
                test_files[fName]={'label_file': aFile, 'raw_main': raw_main, 'raw_llm': raw_llm, 'sceneType': sceneType, 'scene_prob_raw': raw_scene_prob, 'scene_prob_norm': normalized_scene_prob}
                test_files[fName]['annotation']=annot[main_tgt]
                # test_files[fName]['save_dir']=fList.intermediate_save_dir                
                test_files[fName]['cluster_main_base']=fList.intermediate_save_dir+main_tgt.replace(' ','_')
                test_files[fName]['cluster_llm_base']=fList.intermediate_save_dir+llm_tgt.replace(' ','_')
                if classifier_type is not None:
                    test_files[fName]['cluster_main_base']+="."+classifier_type
                    test_files[fName]['cluster_llm_base']+="."+classifier_type
                if is_pose_filtered:
                    test_files[fName]['cluster_main_base']+=".flt"
                    test_files[fName]['cluster_llm_base']+=".flt"
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

import time
TIME_STRUCT={'count': 0, 'times':np.zeros((2,),dtype=float)}

def create_object_clusters(pts_xyz, 
                            pts_prob, 
                            floor_threshold=-0.1, 
                            detection_threshold=0.5, 
                            min_cluster_points=ABSOLUTE_MIN_CLUSTER_SIZE, 
                            gridcell_size=0.01,
                            compress_clusters=True):
    if pts_xyz.shape[0]==0 or pts_xyz.shape[0]!=pts_prob.shape[0]:
        return []
    global TIME_STRUCT
    t_array=[]
    t_array.append(time.time())
    whichP=(pts_prob>=detection_threshold)
    pcd=o3d.geometry.PointCloud()
    xyzF=pts_xyz[whichP]
    F2=np.where(np.isnan(xyzF).sum(1)==0)
    xyzF2=xyzF[F2]        
    pcd.points=o3d.utility.Vector3dVector(xyzF2)
    dbscan_eps=2.4*gridcell_size
    object_clusters=get_distinct_clusters(pcd, 
                                            floor_threshold=floor_threshold, 
                                            cluster_min_count=min_cluster_points,
                                            gridcell_size=gridcell_size,
                                            eps=dbscan_eps)
    t_array.append(time.time())

    probF=pts_prob[whichP]
    probF2=probF[F2]
    for idx in range(len(object_clusters)):
        object_clusters[idx].estimate_probability(xyzF2,probF2)
        if compress_clusters:
            object_clusters[idx].compress_object()
    t_array.append(time.time())
    TIME_STRUCT['times']=TIME_STRUCT['times']+np.diff(np.array(t_array))
    TIME_STRUCT['count']+=1
    if TIME_STRUCT['count']%100==0:
        print(f"******* TIME(create_object_clusters) - {TIME_STRUCT['count']} ****** ")
        print(TIME_STRUCT['times']/500)
        TIME_STRUCT['times']=np.zeros((2,),dtype=float)
        # pdb.set_trace()

    return object_clusters

def get_clusters(save_file, raw_pts, detection_threshold, min_cluster_points):

    if os.path.exists(save_file):
        with open(save_file, 'rb') as handle:
            all_objects=pickle.load(handle)
    else:
        with open(raw_pts,'rb') as handle:
            pcl_raw=pickle.load(handle)
        all_objects=create_object_clusters(pcl_raw['xyz'],pcl_raw['probs'], -1.0, detection_threshold, min_cluster_points=ABSOLUTE_MIN_CLUSTER_SIZE)
        with open(save_file, 'wb') as handle:
            pickle.dump(all_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    objects_out=[]
    for obj in all_objects:
        if obj.size()>min_cluster_points:
            objects_out.append(obj)
    return objects_out

def combine_matches(test_file_dict, detection_threshold=0.5, min_cluster_points=10000):
    try:
        save_main=test_file_dict['cluster_main_base']+".%0.2f.cl.pkl"%(detection_threshold)
        save_llm=test_file_dict['cluster_llm_base']+".%0.2f.cl.pkl"%(detection_threshold)
        objects_main=get_clusters(save_main, test_file_dict['raw_main'], detection_threshold, min_cluster_points)
        objects_llm=get_clusters(save_llm, test_file_dict['raw_llm'], detection_threshold, min_cluster_points)
    except Exception as e:
        print(f"Exception: {e}")
        pdb.set_trace()
        return [], [] ,[]

    if DEBUG_ and len(test_file_dict['annotation'])>0:
        pdb.set_trace()

    positive_clusters=[]
    negative_clusters=[]
    matches=np.zeros((len(objects_main),len(test_file_dict['annotation'])),dtype=int)
    for idx0 in range(len(objects_main)):
        # Match with other clusters
        cl_stats=[idx0, objects_main[idx0].prob_stats['max'], objects_main[idx0].prob_stats['mean'], -1, -1, test_file_dict['scene_prob_raw'], test_file_dict['scene_prob_norm']]
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

    # if DEBUG_ and len(test_file_dict['annotation'])>0:
    if DEBUG_ and len(objects_main)>0:
        pdb.set_trace()            
    return matches, positive_clusters, negative_clusters

def estimate_likelihood(cluster_stats, category):
    if category=='main-max':
        return cluster_stats[1]
    elif category=='main-mean':
        return cluster_stats[2]
    elif category=='llm-max':
        return cluster_stats[3]
    elif category=='llm-mean':
        return cluster_stats[4]
    elif category=='room-raw':
        return cluster_stats[5]
    elif category=='room-norm':
        return cluster_stats[6]
    elif category=='combo-max': # combined max
        return cluster_stats[1]*cluster_stats[3]
    elif category=='combo-mean': # combined mean
        return cluster_stats[2]*cluster_stats[4]
    elif category=='main-room': # combined main + room
        return cluster_stats[2]*cluster_stats[6]
    elif category=='llm-room': # combined main + room
        return cluster_stats[4]*cluster_stats[6]
    elif category=='combo-room': # combined main + room
        return cluster_stats[2]*cluster_stats[4]*cluster_stats[6]
    else:
        raise(Exception(f"Category {category} not implemented yet"))

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
        m2=matches
        for dim_ in range(m2.shape[1]):
            m2[:,dim_]*=positive_clusters
        unmatched_annotations=(m2.sum(0)<1).sum()
        count_true_negative_clusters=((~positive_clusters)*(~whichP)).sum()

    stats[0]+=matched_positive_clusters #TP
    stats[1]+=unmatched_annotations     #FN
    stats[2]+=count_positive_clusters - matched_positive_clusters #FP
    stats[3]+=count_true_negative_clusters

    return stats        
    
def check_top_n(positive_clusters, negative_clusters, category, howmany=1):
    P=np.array([estimate_likelihood(pp, category) for pp in positive_clusters])
    N=np.array([estimate_likelihood(nn, category) for nn in negative_clusters])
    validP=P[np.where(P>0)]
    validN=N[np.where(N>0)]
    arr=np.hstack((validP,validN))
    if arr.shape[0]==0:
        return 0
    srt=np.argsort(-arr)
    # if any of the top N are positive matches, return true
    if (srt[:howmany]<validP.shape[0]).sum()>0:
        return 1
    return -1

def vary_threshold(test_files, cat_list, fixed_size_threshold):
    thresholds=np.arange(0.5,1.0,0.1)

    recallN = np.zeros((thresholds.shape[0],len(cat_list)))
    precisionN = np.zeros((thresholds.shape[0],len(cat_list)))
    # specificityN = np.zeros((thresholds.shape[0],len(cat_list)))
    AUC=np.zeros(len(cat_list))
    
    recall_top1 = np.zeros((thresholds.shape[0],len(cat_list)))
    precision_top1 = np.zeros((thresholds.shape[0],len(cat_list)))
    specificity_top1 = np.zeros((thresholds.shape[0],len(cat_list)))

    is_positive=np.array([ len(test_files[scene]['annotation'])>0 for scene in test_files])
    count_positive_scenes=is_positive.sum()
    count_negative_scenes=len(test_files)-count_positive_scenes        

    for t_idx in range(len(thresholds)):
        threshold=thresholds[t_idx]
        all_results=dict()
        for scene in test_files.keys():
            print(scene)
            matches, positive_clusters, negative_clusters = combine_matches(test_files[scene],
                                                                            detection_threshold=threshold, 
                                                                            min_cluster_points=fixed_size_threshold)
            print(matches)
            all_results[scene]={'matches': matches, 'positive_clusters': positive_clusters, 'negative_clusters': negative_clusters}


        for cat_idx, cat_ in enumerate(cat_list):
            stats=np.zeros((4),dtype=float)
            for scene in all_results:
                stats=stats + get_graph_stats_per_scene(all_results[scene]['matches'], all_results[scene]['positive_clusters'], all_results[scene]['negative_clusters'], cat_, threshold)
            top_1=np.array([check_top_n(all_results[scene]['positive_clusters'],all_results[scene]['negative_clusters'], cat_, howmany=1) for scene in all_results])
            
            # Scene level stats
            neg1=np.where(top_1==0) # find all scenes discarded
            pos1=np.where(top_1!=0) # find all scenes discarded
            recall_top1[t_idx, cat_idx]=(is_positive[pos1]==1).sum()/count_positive_scenes
            precision_top1[t_idx, cat_idx]=(is_positive[pos1]==1).sum()/(pos1[0].shape[0]+1e-6)
            specificity_top1[t_idx, cat_idx]=(is_positive[neg1]==False).sum()/count_negative_scenes

            # Object level stats
            TP=stats[0]
            FN=stats[1]
            FP=stats[2]
            TN=stats[3]
            precisionN[t_idx, cat_idx]=TP/(TP+FP+1e-6)
            recallN[t_idx, cat_idx]=TP/(TP+FN+1e-6)
            # specificityN[t_idx, cat_idx]=TN/(FP+TN+1e-6)

    return recallN, precisionN, recall_top1, precision_top1, specificity_top1, thresholds

def vary_size(test_files, cat_list, fixed_detection_threshold):
    size_thresholds=np.hstack((np.arange(200,3000,200), np.arange(3000,50000,500)))

    recallN = np.zeros((size_thresholds.shape[0],len(cat_list)))
    precisionN = np.zeros((size_thresholds.shape[0],len(cat_list)))
    # specificityN = np.zeros((size_thresholds.shape[0],len(cat_list)))
    AUC=np.zeros(len(cat_list))
    
    recall_top1 = np.zeros((size_thresholds.shape[0],len(cat_list)))
    precision_top1 = np.zeros((size_thresholds.shape[0],len(cat_list)))
    specificity_top1 = np.zeros((size_thresholds.shape[0],len(cat_list)))

    is_positive=np.array([ len(test_files[scene]['annotation'])>0 for scene in test_files])
    count_positive_scenes=is_positive.sum()
    count_negative_scenes=len(test_files)-count_positive_scenes        

    for t_idx in range(len(size_thresholds)):
        size_threshold=size_thresholds[t_idx]
        all_results=dict()
        for scene in test_files.keys():
            # scene='/data3/datasets/scannet/scans/scene0003_01'
            print(scene)
            matches, positive_clusters, negative_clusters = combine_matches(test_files[scene],
                                                                            detection_threshold=fixed_detection_threshold, 
                                                                            min_cluster_points=size_threshold)
            print(matches)
            all_results[scene]={'matches': matches, 'positive_clusters': positive_clusters, 'negative_clusters': negative_clusters}


        for cat_idx, cat_ in enumerate(cat_list):
            stats=np.zeros((4),dtype=float)
            for scene in all_results:
                stats=stats + get_graph_stats_per_scene(all_results[scene]['matches'], all_results[scene]['positive_clusters'], all_results[scene]['negative_clusters'], cat_, fixed_detection_threshold)
            top_1=np.array([check_top_n(all_results[scene]['positive_clusters'],all_results[scene]['negative_clusters'], cat_, howmany=1) for scene in all_results])
            
            # Scene level stats
            neg1=np.where(top_1==0) # find all scenes discarded
            pos1=np.where(top_1!=0) # find all scenes discarded
            recall_top1[t_idx, cat_idx]=(is_positive[pos1]==1).sum()/count_positive_scenes
            precision_top1[t_idx, cat_idx]=(is_positive[pos1]==1).sum()/(pos1[0].shape[0]+1e-6)
            specificity_top1[t_idx, cat_idx]=(is_positive[neg1]==False).sum()/count_negative_scenes

            # Object level stats
            TP=stats[0]
            FN=stats[1]
            FP=stats[2]
            TN=stats[3]
            precisionN[t_idx, cat_idx]=TP/(TP+FP+1e-6)
            recallN[t_idx, cat_idx]=TP/(TP+FN+1e-6)
            # specificityN[t_idx, cat_idx]=TN/(FP+TN+1e-6)
            
    return recallN, precisionN, recall_top1, precision_top1, specificity_top1, size_thresholds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('main_target', type=str, help='main target search string')
    parser.add_argument('llm_target', type=str, help='llm target search string')
    parser.add_argument('--save_dir',type=str,default=None, help='where are the intermediate files saved? By default these are saved to the root dir')
    parser.add_argument('--root_dir',type=str,default='/data3/datasets/scannet/scans/', help='where is the root dir (default = /data3/datasets/scannet/scans/)')
    parser.add_argument('--num_points',type=int,default=None, help='number of points per cluster')
    parser.add_argument('--detection_threshold',type=float,default=None, help='fixed detection threshold')
    parser.add_argument('--classifier_type',type=str,default=None, help='what classifier to use {yolo_world, clipseg} (default=clipseg)')
    parser.add_argument('--pose_filter', dest='pose_filter', action='store_true')
    parser.set_defaults(pose_filter=False)
    args = parser.parse_args()

    # if args.num_points is None and args.detection_threshold is None:
    test_files=build_test_data_structure(args.root_dir,args.main_target, args.llm_target, args.pose_filter, args.classifier_type)
    legend=['main-mean', 'main-room', 'llm-mean', 'llm-room', 'combo-mean', 'combo-room']

    if args.num_points is not None:
        recallN, precisionN, recall_top1, precision_top1, specificity_top1, all_thresholds=vary_threshold(test_files, legend, args.num_points)
    elif args.detection_threshold is not None:
        recallN, precisionN, recall_top1, precision_top1, specificity_top1, all_thresholds=vary_size(test_files, legend, args.detection_threshold)
    else:
        print("Must select either a number of points or a detection threshold")
        import sys
        sys.exit(-1)

    # N=11
    print(legend)
    print("recall_top1")
    print(f"{recall_top1}")
    print("precision_top1")
    print(f"{precision_top1}")
    print("F-score, top1")
    print(f"{2*recall_top1*precision_top1/(recall_top1+precision_top1)}")
    # print("specificity_top1")
    # print(f"{specificity_top1[[0,N],:]}")
    print("recallN")
    print(f"{recallN}")
    print("precisionN")
    print(f"{precisionN}")
    print("F-Score")
    print(f"{2*recallN*precisionN/(recallN+precisionN)}")

    # pdb.set_trace()
    # import matplotlib.pyplot as plt
    # for cat_idx, cat_ in enumerate(legend):
    #     plt.figure(1)
    #     plt.plot(recallN[:,cat_idx], precisionN[:, cat_idx])
    #     plt.figure(2)
    #     tpr=np.hstack((1.0,recall_top1[:,cat_idx]))
    #     fpr=np.hstack((1.0, 1-specificity_top1[:,cat_idx]))
    #     plt.plot(fpr, tpr)
    # plt.figure(1)
    # plt.legend(legend)
    # plt.title('Object Level Recall vs Precision')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.figure(2)
    # plt.legend(legend)
    # plt.title('Scene Level ROC')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

        
