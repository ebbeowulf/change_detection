import os
import sys
import ollama
import argparse
import glob
import pdb
import json
from llm_query_tools import single_level_results_template
import numpy as np

RESULTS_TEMPLATE=single_level_results_template(["object","is_pickup"],[str,bool],["<type of object>", "<True/False>"])
SUMMARY_TEMPLATE=single_level_results_template(["object"],[str],["<describe object in 5 words or less>"])

# TASK_DESCRIPTION = ['We are deciding on tasks for a robot that can pick stuff up and put it away.',
#                     'The robot should  pick up things left behind by people that used the room.'
#                     'This includes all small stuff that does not normally belong in the living room.'
#                     'The owner has requested that the robot not pick up books or other objects related to their work.'
#                     'The object surrounded by the red box in the provided image(s) has been identified by the robot as a candidate for cleaning.',
#                     'Is this an object that the robot should pick up and put away?']
TASK_DESCRIPTION = ['The object surrounded by the blue box in the provided image(s) has been identified as an object that may not belong in this room.',
                    'It is the same object in all provided images.'
                    'Identify the object and state whether the object should be picked up and put away.']
PROMPT=""
for task in TASK_DESCRIPTION:
    PROMPT+=task
PROMPT+="Return an answer in JSON format as " + RESULTS_TEMPLATE.generate_format_prompt()

NUM_MULTI_IMAGES=4

def run_single_image_inference(model: str, image_path: str, print_output: bool=False):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": [image_path]}],
        stream=True,
    )
    message=""
    for chunk in stream:
        message+=chunk["message"]["content"]
        if print_output:
            print(chunk["message"]["content"], end="", flush=True)
    
    return RESULTS_TEMPLATE.recover_json(message)

def run_multi_image_inference(model: str, image_path_list: list, print_output: bool=False):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": image_path_list}],
        stream=True,
    )
    message=""
    for chunk in stream:
        message+=chunk["message"]["content"]
        if print_output:
            print(chunk["message"]["content"], end="", flush=True)
    
    return RESULTS_TEMPLATE.recover_json(message)

def summarize_likely_object(model: str, descriptions:list):
    SUM_PROMPT="I have an object that has been described as all of the following: "
    for desc in descriptions[:-1]:
        SUM_PROMPT+=desc+", "
    SUM_PROMPT+=desc + ". Make a guess as to what this object is and return in JSON format as " + SUMMARY_TEMPLATE.generate_format_prompt()
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": SUM_PROMPT, "images": []}],
        stream=True,
    )
    message=""
    for chunk in stream:
        message+=chunk["message"]["content"]  
    return SUMMARY_TEMPLATE.recover_json(message)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_image_dir',type=str,help='location of cluster images to be processed')
    parser.add_argument('prefix',type=str,help='images should of form <prefix>_<cluster_id>_<image_id>.png. What is the prefix?')
    parser.add_argument('--model-name',type=str,default='llama4:scout',help='type of ollama model to run. Default = llama4:scout')
    parser.add_argument('--no_change', dest='use_change', action='store_false')
    parser.set_defaults(use_change=True)
    args=parser.parse_args()

    valid_models = ["llava:13b", "llava-llama3", "llama4:scout", "moondream"]
    if args.model_name not in valid_models:
        print(f"Error: Invalid model name. Choose from: {', '.join(valid_models)}")
        sys.exit(1)

    all_results=[]
    for cluster_idx in range(20):
        capture_results=dict()
        if args.use_change:
            all_image_files=glob.glob(f"{args.cluster_image_dir}/{args.prefix}_{cluster_idx}*.png")
        else:
            all_image_files=glob.glob(f"{args.cluster_image_dir}/{args.prefix}_{cluster_idx}*.OV.png")
        print(f"Found {len(all_image_files)} in cluster {cluster_idx}")
        if len(all_image_files)>0:
            all_image_files=np.array(all_image_files)
            if all_image_files.shape[0]<=NUM_MULTI_IMAGES:
                capture_results[0]=run_multi_image_inference(args.model_name, all_image_files.tolist())
            else:
                num_runs=int(np.ceil(all_image_files.shape[0]/NUM_MULTI_IMAGES))
                for i in range(num_runs):
                    capture_results[i]=run_multi_image_inference(args.model_name, np.random.choice(all_image_files,NUM_MULTI_IMAGES).tolist())
            # pdb.set_trace()
            # for image_name in all_image_files:
            #     capture_results[image_name]=run_single_image_inference(args.model_name, image_name)
        all_results.append(capture_results)
        
    import pickle
    if args.use_change:
        save_file=args.cluster_image_dir+"/"+args.prefix+".llm.pkl"
    else:
        save_file=args.cluster_image_dir+"/"+args.prefix+".OV.llm.pkl"
    with open(save_file,'wb') as fout:
        pickle.dump(all_results,fout)


if __name__ == "__main__":
    main()

