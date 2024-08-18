import json
import pdb
from pcloud_models.rgbd_file_list import rgbd_file_list

def build_test_data_structure(root_dir:str, main_tgt:str, llm_tgt:str):
    import glob
    import os
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
                test_files[fName]={'label_file': aFile}
                test_files[fName]={'raw_main': raw_main}
                test_files[fName]={'raw_llm': raw_llm}
                test_files[fName]['annotation']=annot[main_tgt]
    return test_files

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('main_tgt',type=str,help='main target query (i.e. backpack)')
    parser.add_argument('llm_tgt',type=str,help='llm replacement query (i.e. "strap-on bag for carrying")')
    args = parser.parse_args()

    pdb.set_trace()
    build_test_data_structure()