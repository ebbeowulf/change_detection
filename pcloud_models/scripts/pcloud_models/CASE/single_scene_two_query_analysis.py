import numpy as np
from two_query_localize import FIXED_ROOM_STATS_FILE, combine_matches
import json
from rgbd_file_list import rgbd_file_list
import sys
from scannet_processing import get_scene_type
import os

def build_test_data_structure(fName:str, main_tgt:str, llm_tgt:str, is_pose_filtered=False):
    with open(FIXED_ROOM_STATS_FILE, 'r') as fin:
        all_room_probs=json.load(fin)
    if main_tgt not in all_room_probs:
        print(f"No room probabilities found for {main_tgt}")
        sys.exit(-1)
    room_prob=all_room_probs[main_tgt]
    max_room_prob=max([room_prob[room] for room in room_prob])

    #We also want the scene type from params file
    s_root=fName.split('/')
    if s_root[-1]=='':
        par_file=fName+"%s.txt"%(s_root[-2])
    else:
        par_file=fName+"/%s.txt"%(s_root[-1])        
    sceneType=get_scene_type(par_file)

    fList=rgbd_file_list(fName+"/raw_output",fName+"/raw_output",fName+"/raw_output/save_results", is_pose_filtered)
    aFile=fList.get_annotation_file()
    raw_main=fList.get_combined_raw_fileName(main_tgt)
    raw_llm=fList.get_combined_raw_fileName(llm_tgt)
    test_files=dict()
    
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
            test_files={'label_file': aFile, 'raw_main': raw_main, 'raw_llm': raw_llm, 'sceneType': sceneType, 'scene_prob_raw': raw_scene_prob, 'scene_prob_norm': normalized_scene_prob}
            test_files['annotation']=annot[main_tgt]
            # test_files[fName]['save_dir']=fList.intermediate_save_dir                
            test_files['cluster_main_base']=fList.intermediate_save_dir+main_tgt.replace(' ','_')
            test_files['cluster_llm_base']=fList.intermediate_save_dir+llm_tgt.replace(' ','_')
            if is_pose_filtered:
                test_files['cluster_main_base']+=".flt"
                test_files['cluster_llm_base']+=".flt"
    return test_files

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('main_target', type=str, help='main target search string')
    parser.add_argument('llm_target', type=str, help='llm target search string')
    parser.add_argument('--num_points',type=int,default=200, help='number of points per cluster')
    parser.add_argument('--detection_threshold',type=float,default=0.5, help='fixed detection threshold')
    parser.add_argument('--pose_filter', dest='pose_filter', action='store_true')
    parser.set_defaults(pose_filter=False)
    args = parser.parse_args()

    test_file_dict=build_test_data_structure(args.root_dir, args.main_target, args.llm_target, args.pose_filter)
    combine_matches(test_file_dict, detection_threshold=args.detection_threshold, min_cluster_points=args.num_points)