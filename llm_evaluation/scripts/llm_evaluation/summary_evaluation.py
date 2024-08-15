# import open3d as o3d
import os
import json
import glob
from pcloud_models.rgbd_file_list import rgbd_file_list
import pdb
import numpy as np
import json

ROOT_DIR="/home/emartinso/data/scannet/scans/"
TGT_CLASS='backpack'
IOU_THRESHOLD=0.05

# EVAL_FILE_TYPE="summary_file"
EVAL_FILE_TYPE="llama_file"

fixed_room_prob={
  "Living room / Lounge": 0.6,
  "Bathroom": 0.1,
  "Office": 0.7,
  "Bookstore / Library": 0.4,
  "Lobby": 0.3,
  "Conference Room": 0.2,
  "Bedroom / Hotel": 0.8,
  "ComputerCluster": 0.5,
  "Copy/Mail Room": 0.3,
  "Hallway": 0.7,
  "Closet": 0.9,
  "Kitchen": 0.2,
  "Classroom": 0.9,
  "Apartment": 0.8,
  "Dining Room": 0.3,
  "Storage/Basement/Garage": 0.7,
  "Stairs": 0.2
}

def get_llama_summary_fileName(fList, cls:str):
    return fList.intermediate_save_dir+"%s.nvidia_llama.room.json"%(cls)    

def build_test_data_structure(root_dir:str, tgt_object:str, eval_file_type=EVAL_FILE_TYPE):
    # Find all files with the '.txt' extension in the current directory
    txt_files = glob.glob(root_dir+"/*")
    test_files=dict()
    for fName in txt_files:
        fList=rgbd_file_list(fName+"/raw_output",fName+"/raw_output",fName+"/raw_output/save_results")
        aFile=fList.get_annotation_file()
        if eval_file_type=="summary_file":
            summaryFile=fList.get_json_summary_fileName(tgt_object)
        else:
            summaryFile=get_llama_summary_fileName(fList, tgt_object)

        if os.path.exists(aFile) and os.path.exists(summaryFile):
            with open(aFile) as fin:
                annot=json.load(fin)
            if tgt_object in annot:
                test_files[fName]={'label_file': aFile, 'summary_file': summaryFile}
                test_files[fName]['annotation']=annot[tgt_object]
            # else:
            #     test_files[fName]['annotation']=annot[tgt_object]
    return test_files

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
#      number of matched clusters    [tp] 
#      number of unmatched clusters  [fp] 
#      number of unmatched annotations [fn]
def compare_annotations(annotations, cluster_boxes):
    if cluster_boxes is None:
        return [0,0,0,len(annotations)]

    num_clusters=len(cluster_boxes)

    if len(annotations)<1:
        return [0,0,num_clusters,0]

    matchA=np.zeros((len(annotations)),dtype=bool)
    matchC=np.zeros(num_clusters,dtype=bool)
    for cl_idx, cluster in enumerate(cluster_boxes):
        box_min=cluster[0]
        box_max=cluster[1]
        for a_idx, annot in enumerate(annotations):
            iou=calculate_iou(box_min, box_max, np.array(annot[:3]),np.array(annot[3:]))
            if iou>IOU_THRESHOLD:
                matchA[a_idx]=True
                matchC[cl_idx]=True
                break

    # if matchC.sum()==0:
    #     pdb.set_trace()
    return [matchC.sum(),matchA.sum(),(matchC==False).sum(),(matchA==False).sum()]

# Calculate the maximum IoU between the box and all known annotations
def match_box_to_annotations(box, annotations):
    if len(annotations)<1:
        return 0.0

    if len(box)==6:
        box_min=box[:3]
        box_max=box[3:]
    else:
        box_min=box[0]
        box_max=box[1]
    max_iou=0.0
    for a_idx, annot in enumerate(annotations):
        iou=calculate_iou(box_min, box_max, np.array(annot[:3]),np.array(annot[3:]))
        if iou>max_iou:
            max_iou=iou
    return max_iou

# def extract_matches(test_files:dict, save_file:str, method):
#     # try:
#     #     with open(save_file,'r') as fin:
#     #         results=json.load(fin)
#     # except Exception as e:
#     #     results=dict()

#     all_room_prob=[]
#     room_list=[]
#     for key in test_files:
#         print(key)
#         results[key]={'threshold': [], 'matches': []}
#         try:
#             with open(test_files[key]['summary_file'], 'r') as fin:
#                 summary=json.load(fin)
#         except Exception as e:
#             print(f"Error loading summary file: {e}")
#             pdb.set_trace()

#         cl_boxes=np.array(summary['object_results']['0.50']['boxes'])
#         prob=np.array(summary['object_results']['0.50']['mean_prob'])

#         room_prob=get_room_prob(summary)
#         all_room_prob.append(room_prob)
#         if room_prob>=0.0:
#             # rescale from [0.0,0.6] to [0.5,1.0]
#             # rp=(min(0.5,max(0.1,room_prob))-0.1)/0.4
#             rp=(min(0.5,room_prob))/0.5
#             room_prob = 0.5*rp + 0.5
#             # room_prob = rp
#         else:
#             room_prob=0.75
        
#         roomT=summary['room'].replace('I am in a ','')[:-1]
#         # room_list.append(roomT[:-1])
#         if roomT in fixed_room_prob:
#             f_room_prob=fixed_room_prob[roomT]
#         else:
#             pdb.set_trace()
#         print(f"Room Probability = {room_prob}")

#         for threshold in np.arange(0.1,1.0,0.02):
#             # matches={}
#             whichB=(prob>=threshold)
#             if method==0:
#                 matches=compare_annotations(test_files[key]['annotation'],cl_boxes[whichB])
#             elif method ==1:
#                 whichB=((prob*room_prob)>=threshold)
#                 matches=compare_annotations(test_files[key]['annotation'],cl_boxes[whichB])
#             elif method ==2:
#                 whichB=((prob*f_room_prob)>=threshold)
#                 matches=compare_annotations(test_files[key]['annotation'],cl_boxes[whichB])
#             else:
#                 print("Not a valid method")
#                 continue
            
#             # matches=compare_annotations(test_files[key]['annotation'],cl_boxes[whichB])
#             # matches['baseline']=compare_annotations(test_files[key]['annotation'],cl_boxes[whichB])

#             results[key]['threshold'].append(threshold)
#             results[key]['matches'].append(matches)

#     # all_rooms=np.unique(room_list)
#     # for room in all_rooms:
#     #     print(room)
#     # pdb.set_trace()
#     return results

def estimate_per_object_stats(test_files:dict, method):    
    positive=[]
    negative=[]
    for key in test_files:
        print(key)
        try:
            with open(test_files[key]['summary_file'], 'r') as fin:
                summary=json.load(fin)
        except Exception as e:
            print(f"Error loading summary file: {e}")
            # pdb.set_trace()       

        # Calculuate room probabilities
        room_prob=get_room_prob(summary)
        if room_prob<0:
            room_prob=0.75
        
        roomT=summary['room'].replace('I am in a ','')[:-1]
        if roomT in fixed_room_prob:
            f_room_prob=fixed_room_prob[roomT]
        else:
            pdb.set_trace()
        print(f"Room Probability = {room_prob}")        

        num_boxes = len(summary['object_results']['0.50']['boxes'])
        for b_idx in range(num_boxes):
            box=np.array(summary['object_results']['0.50']['boxes'][b_idx])
            iou=match_box_to_annotations(box,         test_files[key]['annotation'])
            if method==0:   # max prob
                lk=summary['object_results']['0.50']['max_prob'][b_idx]
            elif method==1: # mean prob
                lk=summary['object_results']['0.50']['mean_prob'][b_idx]
            elif method==2: # room only
                lk=room_prob
            elif method==3: # fixed room only
                lk=f_room_prob
            elif method==4: # combine room + object prob
                lk=room_prob * 0.3 + 0.7*summary['object_results']['0.50']['mean_prob'][b_idx]
            elif method==5: # combine room + object prob
                lk=f_room_prob * 0.3 + 0.7* summary['object_results']['0.50']['mean_prob'][b_idx]
            else:
                print("Method not found - exiting")
                continue
            if iou>IOU_THRESHOLD:
                positive.append(lk)
            else:
                negative.append(lk)
    positive=np.array(positive)
    negative=np.array(negative)
    threshold_range=np.arange(0.1,1.0,0.03)
    results={'tp': np.zeros(threshold_range.shape,dtype=int),'fp': np.zeros(threshold_range.shape,dtype=int),'fn': np.zeros(threshold_range.shape,dtype=int),'tn': np.zeros(threshold_range.shape,dtype=int)}
    for t_idx, threshold in enumerate(threshold_range):
        results['tp'][t_idx]=(positive>threshold).sum()
        results['fn'][t_idx]=positive.shape[0]-results['tp'][t_idx]
        results['fp'][t_idx]=(negative>threshold).sum()
        results['tn'][t_idx]=negative.shape[0]-results['fp'][t_idx]

    return results

def get_room_prob(map_summary):
    ss=map_summary['llama']['room']['room+furniture']['output']
    bracket1=[pos for pos, char in enumerate(ss) if char == '{']
    bracket2=[pos for pos, char in enumerate(ss) if char == '}']
    # pdb.set_trace()
    for bI1 in bracket1:
        for bI2 in bracket2:
            substr=ss[bI1:(bI2+1)]
            substr=substr.replace('%','')
            try:
                A=json.loads(substr)
                all_val=[ A[key] for key in A.keys() ]
                if len(all_val)==1:
                    # is this a percentage?
                    # if type(all_val[0])==str and all_val[0][-1]=='%':
                    #     numval=float(all_val[0][:-1])
                    # else:
                    numval=float(all_val[0])
                    if numval > 1.0 and numval <=100.0:
                        return numval / 100.0
                    return numval
            except Exception as e:
                # not a valid JSON substr - try another pair
                continue
    return -1


# def eval(all_results):
#     all_stats=dict()
#     pdb.set_trace()
#     for method in all_results.keys():
#         res_hash=dict()
#         results=all_results[method]
#         for key in results:
#             try:
#                 if 'threshold' not in results[key] or 'matches' not in results[key]:
#                     continue
#                 for threshold, match in zip(results[key]['threshold'],results[key]['matches']):
#                     K1=int(threshold*100)
#                     if K1 not in res_hash:
#                         res_hash[K1]=[]
#                     res_hash[K1].append(match)
#             except Exception as e:
#                 pdb.set_trace()

#         a_s=[]
#         max_cl_count=0
#         for thresh in res_hash:
#             cum_sum=np.array(res_hash[thresh]).sum(0)
#             if max_cl_count<cum_sum[2]:
#                 max_cl_count=cum_sum[2]
#             if (cum_sum[0]+cum_sum[2])==0:
#                 precision=0.0
#             else:
#                 precision=cum_sum[0]/(cum_sum[0]+cum_sum[2])
#             recall=cum_sum[1]/(cum_sum[1]+cum_sum[3])
#             a_s.append([thresh,precision,recall,cum_sum[2]])
#         # Calculate specificity post-hoc based on maximum number of 
#         #   false positives
#         a_s=np.array(a_s)
#         a_s[:,3] = (max_cl_count-a_s[:,3])/max_cl_count
#         all_stats[method]=a_s

#     # Calculate s
#     import matplotlib.pyplot as plt
#     plt.figure()
#     for method in all_stats:        
#         a_s=np.array(all_stats[method])
#         plt.plot(a_s[:,2],a_s[:,1])
#     plt.title('Precision vs Recall')
#     plt.legend(list(all_stats.keys()))

#     plt.figure()
#     for method in all_stats:        
#         a_s=np.array(all_stats[method])
#         plt.plot(a_s[:,3],a_s[:,1])
#     plt.title('Precision vs Specificity')
#     plt.legend(list(all_stats.keys()))

#     # for thresh in res_hash:
#     #     pdb.set_trace()
#     #     F=np.where(all_stats[0]==thresh)
#     #     plt.plot(all_stats[F][:,3],all_stats[F][:,2])
#     plt.show()

if __name__ == '__main__':
    test_files=build_test_data_structure(ROOT_DIR,TGT_CLASS)
    results={}
    R0=estimate_per_object_stats(test_files, method=0)
    R1=estimate_per_object_stats(test_files, method=1)
    R2=estimate_per_object_stats(test_files, method=2)
    R3=estimate_per_object_stats(test_files, method=3)
    R4=estimate_per_object_stats(test_files, method=4)
    R5=estimate_per_object_stats(test_files, method=5)
    all_results={'maxP': R0, 
             'meanP': R1,
             'roomP': R2,
             'f_roomP': R3,
             'room+mean': R4,
             'f_room+mean': R5}

    import matplotlib.pyplot as plt
    legend=[]
    for key in all_results:
        results=all_results[key]
        prec=results['tp']/(results['tp']+results['fp']+1e-4)
        recall=results['tp']/(results['tp']+results['fn'])
        spec=results['tn']/(results['tn']+results['fp'])
        legend.append(key)
        plt.figure(1)
        plt.plot(spec, prec)
        plt.figure(2)
        plt.plot(recall, prec)
        plt.figure(3)
        plt.plot(1-spec, recall)
    plt.figure(1)
    plt.xlabel('Specificity')
    plt.ylabel('Precision')
    plt.legend(legend)
    plt.title('Precision vs Specificity')
    plt.figure(2)
    plt.legend(legend)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.figure(3)
    plt.legend(legend)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.show()

    # pdb.set_trace()    
    # R0=extract_matches(test_files,TGT_CLASS+".json",method=0)
    # R1=extract_matches(test_files,TGT_CLASS+".json",method=1)
    # R2=extract_matches(test_files,TGT_CLASS+".json",method=2)
    # pdb.set_trace()
    # results={'baseline': R0, 'room': R1, 'f_room': R2}
    # eval(results)
    pdb.set_trace()
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
    
