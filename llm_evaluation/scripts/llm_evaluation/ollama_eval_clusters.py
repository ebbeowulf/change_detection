import os
import sys
import ollama
import argparse
import glob
import pdb

TASK_DESCRIPTION = ['We are deciding on tasks for a robot that can pick stuff up and put it away.',
                    'Do not pick up stuff that is currently being read or worked on',
                    'The object surrounded by the red box in the provided image has been identified by the robot as a candidate for cleaning.' 
                    'Is it an object that the robot should pick up and put away?'
                    'Return an answer in JSON format as {"object": <type of object>, "pickup": <yes/no>}']
PROMPT=""
for task in TASK_DESCRIPTION:
    PROMPT+=task

def run_single_image_inference(model: str, image_path: str):
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT, "images": [image_path]}],
        stream=True,
    )
    message=""
    for chunk in stream:
        message+=chunk["message"]["content"]
        print(chunk["message"]["content"], end="", flush=True)
    
    pdb.set_trace()

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

    for cluster_idx in range(20):
        all_image_files=glob.glob(f"{args.cluster_image_dir}/{args.prefix}_{cluster_idx}*.png")
        print(f"Found {len(all_image_files)} in cluster {cluster_idx}")
        if len(all_image_files)>0:
            for image_name in all_image_files:
                run_single_image_inference(args.model_name, image_name)


if __name__ == "__main__":
    main()

