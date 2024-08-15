import json
from summary_evaluation import get_room_prob
import pdb
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
# ROOT_DIR="/home/emartinso/data/scannet/scans/"
# TGT_CLASS='backpack'
# IOU_THRESHOLD=0.05

# def extract_room_match_stats(test_files:dict, method):
#     all_room_prob=[]
#     room_list=[]
#     threshold_range=np.arange(0.1,1.0,0.03)
#     results={'tp': np.zeros(threshold_range.shape,dtype=int),'fp': np.zeros(threshold_range.shape,dtype=int),'fn': np.zeros(threshold_range.shape,dtype=int),'tn': np.zeros(threshold_range.shape,dtype=int)}

#     for key in test_files:
#         print(key)
#         try:
#             with open(test_files[key]['summary_file'], 'r') as fin:
#                 summary=json.load(fin)
#         except Exception as e:
#             print(f"Error loading summary file: {e}")
#             pdb.set_trace()

#         room_prob=np.random.rand(1)
#         if method==1:
#             room_prob=get_room_prob(summary)
#             if room_prob < 0:
#                 room_prob=0.5
#         elif method==2:
#             roomT=summary['room'].replace('I am in a ','')[:-1]
#             if roomT in fixed_room_prob:
#                 f_room_prob=fixed_room_prob[roomT]
#             else:
#                 pdb.set_trace()
#         print(f"Room Probability = {room_prob}")

#         for t_idx, threshold in enumerate(threshold_range):
#             if len(test_files[key]['annotation'])>0: # is there a real object in the room?
#                 if threshold<=room_prob:
#                     results['tp'][t_idx]+=1
#                 else:
#                     results['fn'][t_idx]+=1
#             else:
#                 if threshold<=room_prob:
#                     results['fp'][t_idx]+=1
#                 else:
#                     results['tn'][t_idx]+=1
#     return results

# test_files=build_test_data_structure(ROOT_DIR,TGT_CLASS)
# R0=extract_room_match_stats(test_files, method=0)
# R1=extract_room_match_stats(test_files, method=1)
# R2=extract_room_match_stats(test_files, method=2)
# results={'baseline': R0, 'room': R1, 'f_room': R2}

# legend=[]
# for key in results:
#     prec=results[key]['tp']/(results[key]['tp']+results[key]['fp']+1e-4)
#     recall=results[key]['tp']/(results[key]['tp']+results[key]['fn'])
#     legend.append(key)
#     plt.plot(recall,prec)

# plt.title('Precision vs Recall')
# plt.legend(legend)
# plt.show()

def get_all_max_object_lk(object_type):
    ROOT_DIR="/home/emartinso/data/scannet/scans/"
    test_files=build_test_data_structure(ROOT_DIR,object_type)
    all_object_lk=dict()
    for key in test_files:
        try:
            with open(test_files[key]['summary_file'], 'r') as fin:
                summary=json.load(fin)
        except Exception as e:
            print(f"Error loading summary file: {e}")
        objectP=summary['object_results']['0.50']['mean_prob']
        if len(objectP)>0:
            all_object_lk[key]=max(objectP)
        else:
            all_object_lk[key]=0.5-1e-6
    return all_object_lk

def extract_room_match_stats(llm_result, objects_per_scene, room_type_per_scene, object_type, object_lk, method=0):
    threshold_range=np.arange(0.1,1.0,0.03)
    results={'tp': np.zeros(threshold_range.shape,dtype=int),'fp': np.zeros(threshold_range.shape,dtype=int),'fn': np.zeros(threshold_range.shape,dtype=int),'tn': np.zeros(threshold_range.shape,dtype=int)}

    all_room_types=[]
    all_positive=[]
    all_negative=[]

    for scene in objects_per_scene:
        try:
            lk=np.random.rand(1) # random baseline
            roomT=room_type_per_scene[scene]
            if roomT not in all_room_types:
                all_room_types.append(roomT)

            if method==1: # llm result
                lk=llm_result[roomT]
            elif method==2: # object_lk
                if scene in object_lk:
                    lk=object_lk[scene]
                else:
                    continue
            elif method==3: # object_lk
                if scene in object_lk:
                    lk=object_lk[scene]*0.1 + llm_result[roomT]*0.9
                else:
                    continue
        except Exception as e:
            print(f"Skipping {scene} due to extraction error")
            continue

        for t_idx, threshold in enumerate(threshold_range):
            if object_type in objects_per_scene[scene]:
                all_positive.append(lk)
                if threshold<=lk:
                    results['tp'][t_idx]+=1
                else:
                    results['fn'][t_idx]+=1
            else:
                all_negative.append(lk)
                if threshold<=lk:
                    results['fp'][t_idx]+=1
                else:
                    results['tn'][t_idx]+=1

        # if object_type in objects_per_scene[scene]:
        #     pdb.set_trace()
    
    print("Set of detected room types:")
    for roomT in all_room_types:
        print(roomT, end=", ")
    return results

def trapz(y,x):
    ym=np.mean(np.vstack((y[:-1],y[1:])),0)
    width=np.abs(np.diff(x))
    return (width*ym).sum()

def build_test_data_structure(root_dir:str, tgt_object:str):
    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(root_dir+"/*")
    test_files=dict()
    sumFName=f"/raw_output/save_results/{tgt_object}.summary.json"
    sumFName=sumFName.replace(' ','_')
    for fName in txt_files:
        summaryFile=fName+sumFName
        # pdb.set_trace()
        if os.path.exists(summaryFile):
            test_files[fName]={'summary_file': summaryFile}
    return test_files

if __name__ == '__main__':
    FIXED_ROOM_STATS_FILE="/home/emartinso/data/scannet/scans/llm_likelihoods.json"
    OBJECT_IN_ROOM_STATS_FILE="/home/emartinso/data/scannet/scans/object_in_room_stats.json"

    with open(FIXED_ROOM_STATS_FILE, 'r') as fin:
        llm_result=json.load(fin)

    with open(OBJECT_IN_ROOM_STATS_FILE, 'r') as fin:
        detection_results=json.load(fin)

    legend=[]
    auc=dict()
    for key in llm_result:
        object_lk=get_all_max_object_lk(key)
        auc[key]=[]

        results=extract_room_match_stats(llm_result[key],detection_results['objects_per_scene'],detection_results['room_type_per_scene'],key,object_lk, method=0)
        prec=results['tp']/(results['tp']+results['fp']+1e-4)
        recall=results['tp']/(results['tp']+results['fn'])
        spec=results['tn']/(results['tn']+results['fp'])
        auc[key].append(trapz(recall,1-spec))
        
        # plt.plot(spec,prec)

        results=extract_room_match_stats(llm_result[key],detection_results['objects_per_scene'],detection_results['room_type_per_scene'],key,object_lk,method=1)
        prec=results['tp']/(results['tp']+results['fp']+1e-4)
        recall=results['tp']/(results['tp']+results['fn'])
        spec=results['tn']/(results['tn']+results['fp'])
        auc[key].append(trapz(recall,1-spec))
        # plt.plot(spec,prec)

        results=extract_room_match_stats(llm_result[key],detection_results['objects_per_scene'],detection_results['room_type_per_scene'],key,object_lk,method=2)
        prec=results['tp']/(results['tp']+results['fp']+1e-4)
        recall=results['tp']/(results['tp']+results['fn'])
        spec=results['tn']/(results['tn']+results['fp'])
        auc[key].append(trapz(recall,1-spec))
        # plt.plot(spec,prec)

        results=extract_room_match_stats(llm_result[key],detection_results['objects_per_scene'],detection_results['room_type_per_scene'],key,object_lk,method=3)
        prec=results['tp']/(results['tp']+results['fp']+1e-4)
        recall=results['tp']/(results['tp']+results['fn'])
        spec=results['tn']/(results['tn']+results['fp'])
        auc[key].append(trapz(recall,1-spec))
        # plt.plot(spec,prec)

        # plt.legend(['random','fixed room', 'mean object','combined'])
        # plt.title(key)
        # plt.show()
        # pdb.set_trace()

    pdb.set_trace()
    # plt.title('Precision vs Specificity')
    # # plt.title('Precision vs Recall')
    # plt.legend(legend)
    # plt.show()
