import os
import sys
import ollama
import argparse
import glob
import pdb
import json
from llm_query_tools import single_level_results_template

RESULTS_TEMPLATE=single_level_results_template(["object","is_pickup"],[str,bool],["<type of object>", "<True/False>"])
SUMMARY_TEMPLATE=single_level_results_template(["object"],[str],["<describe object in 5 words or less>"])

TASK_DESCRIPTION = ['We are deciding on tasks for a robot that can pick stuff up and put it away.',
                    'Do not pick up stuff that is currently being read or worked on',
                    'The object surrounded by the red box in the provided image has been identified by the robot as a candidate for cleaning.',
                    'Is it an object that the robot should pick up and put away?']
PROMPT=""
for task in TASK_DESCRIPTION:
    PROMPT+=task
PROMPT+="Return an answer in JSON format as " + RESULTS_TEMPLATE.generate_format_prompt()

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
    args=parser.parse_args()

    valid_models = ["llava:13b", "llava-llama3", "llama4:scout", "moondream"]
    if args.model_name not in valid_models:
        print(f"Error: Invalid model name. Choose from: {', '.join(valid_models)}")
        sys.exit(1)

    all_results=[]
    for cluster_idx in range(20):
        capture_results=dict()
        all_image_files=glob.glob(f"{args.cluster_image_dir}/{args.prefix}_{cluster_idx}*.png")
        print(f"Found {len(all_image_files)} in cluster {cluster_idx}")
        if len(all_image_files)>0:
            for image_name in all_image_files:
                capture_results[image_name]=run_single_image_inference(args.model_name, image_name)
        all_results.append(capture_results)
        
    import pickle
    save_file=args.cluster_image_dir+"/"+args.prefix+".llm.pkl"
    with open(save_file,'wb') as fout:
        pickle.dump(all_results,fout)


if __name__ == "__main__":
    main()

