# import os
# import requests
# import torch
# from PIL import Image
# from transformers import MllamaForConditionalGeneration, AutoProcessor

# model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(model_id)

# # url = "https://cdn.pixabay.com/photo/2017/03/07/22/17/cabin-2125387_1280.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)
# local_path ="/home/emartinso/tmp/backpack/bp_chair.png"
# image = Image.open(local_path)

# messages = [
#     {"role": "user", "content": [
#         {"type": "image"},
#         {"type": "text", "text": "Describe this image in detail."}
#     ]}
# ]
# input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
# inputs = processor(image, input_text, return_tensors="pt").to(model.device)

# output = model.generate(**inputs, max_new_tokens=28000)
# print(processor.decode(output[0]))
  
from ollama_eval_clusters import summarize_likely_object
import pickle
import pdb
import numpy as np
MIN_IMAGES=5

# with open("/data2/datasets/living_room/specAI2/6_24_v1/save_results/decorative_and_functional_items.llm.pkl","rb") as fin:
# with open("/data2/datasets/living_room/specAI2/6_24_v1/save_results/general_clutter.llm.pkl","rb") as fin:
# with open("/data2/datasets/living_room/specAI2/6_24_v1/save_results/floor-level_objects.llm.pkl","rb") as fin:
# with open("/data2/datasets/living_room/specAI2/6_24_v1/save_results/trash_items.llm.pkl","rb") as fin:
with open("/data2/datasets/living_room/specAI2/6_24_v2/save_results/general_clutter.llm.pkl","rb") as fin:
    all_results=pickle.load(fin)

for cluster_idx, capture_results in enumerate(all_results):
    descriptions=[]
    is_pickup=[]
    for key in capture_results.keys():
        if "object" in capture_results[key] and capture_results[key]["object"] is not None:
            descriptions.append(capture_results[key]["object"])
        if "is_pickup" in capture_results[key] and capture_results[key]["is_pickup"] is not None:
            is_pickup.append(capture_results[key]["is_pickup"])
    if len(descriptions)>MIN_IMAGES and len(is_pickup)>MIN_IMAGES:
        object_type=summarize_likely_object("llama4:scout",descriptions)
        pickup_likelihood=np.mean(is_pickup)
        print(f"Cluster {cluster_idx}: '{object_type['object']}', pickup: {pickup_likelihood}")

