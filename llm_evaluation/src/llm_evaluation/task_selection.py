from ollama_eval_clusters import summarize_likely_object
import pickle
import pdb
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('llm_pkl_file',type=str,help='location of the llm.pkl file to process')
    parser.add_argument('--model-name',type=str,default='llama4:scout',help='type of ollama model to run. Default = llama4:scout')
    parser.add_argument('--min-images', type=int, default=5, help='minimum number of images per cluster to consider')
    args=parser.parse_args()


    with open(args.llm_pkl_file,"rb") as fin:
        all_results=pickle.load(fin)

    for cluster_idx, capture_results in enumerate(all_results):
        descriptions=[]
        is_pickup=[]
        for key in capture_results.keys():
            if "object" in capture_results[key] and capture_results[key]["object"] is not None:
                descriptions.append(capture_results[key]["object"])
            if "is_pickup" in capture_results[key] and capture_results[key]["is_pickup"] is not None:
                is_pickup.append(capture_results[key]["is_pickup"])
        if len(descriptions)>args.min_images and len(is_pickup)>args.min_images:
            object_type=summarize_likely_object(args.model_name,descriptions)
            pickup_likelihood=np.mean(is_pickup)
            print(f"Cluster {cluster_idx}: '{object_type['object']}', pickup: {pickup_likelihood}")

