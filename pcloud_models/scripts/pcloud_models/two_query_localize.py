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

ROOT_DIR="/home/ebeowulf/data/scannet/scans/"
IOU_THRESHOLD=0.05
MAIN_TARGET="backpack"
LLM_TARGET="strap-on bag for carrying"

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

def eval_matches(test_file_dict):
    pdb.set_trace()
    try:
        with open(test_file_dict['raw_main'],'rb') as handle:
            pcl_main=pickle.load(handle)
        with open(test_file_dict['raw_llm'],'rb') as handle:
            pcl_llm=pickle.load(handle)
        if pcl_main is None:
            return None
        if pcl_llm is None:
            return None
    except Exception as e:
        return None

    pdb.set_trace()

if __name__ == '__main__':
    test_files=build_test_data_structure(ROOT_DIR,MAIN_TARGET, LLM_TARGET)

    for scene in test_files.keys():
        eval_matches(test_files[scene])
