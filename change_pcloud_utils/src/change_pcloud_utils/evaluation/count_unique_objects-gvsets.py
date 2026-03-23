import json
import os
from evaluation_utils import get_tgt_dir
import pdb

# List of your JSON filenames
# directories=['save_results_unfiltered/sam3_0.1_change','save_results_unfiltered/sam3_0.1_openVocab','save_results_unfiltered/clipseg_0.1_change','save_results_unfiltered/clipseg_0.1_openVocab']
# prompts = ['clothing','dishes','general_clutter','small_items']
directories=['save_results_unfiltered/sam3_0.1_openVocab']
prompts = ['electronics','trash','general_clutter','small_items']

tgt_dir_list=get_tgt_dir('s120')

count={prompt: {'unchanged': 0, 'changed': 0} for prompt in prompts}

for tgt_dir in tgt_dir_list:
    suffix='.labels.tight.json'
    for d in directories:
        for prompt in prompts:
            fileName=tgt_dir+d+"/"+prompt+suffix
            # path = os.path.join(target, d, prompt,suffix)
            with open(fileName, "r") as f:
                data = json.load(f)
            
            for item in data.values():
                if item["name"]=="unchanged":
                    count[prompt]['unchanged']+=1
                else:
                    count[prompt]['changed']+=1
print(count)

