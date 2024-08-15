import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import json
import sys
import pdb
from query_generation import object_likelihood, room_likelihood

def get_llm_response(quantized_model, tokenizer, input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = quantized_model.generate(**input_ids, max_new_tokens=200)
    sout=tokenizer.decode(output[0], skip_special_tokens=True)
    return sout

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('map_summary',type=str,help='full path to json map summary file')
    parser.add_argument('tgt_class',type=str,help='type of object being detected')    
    parser.add_argument('save_file',type=str,help='full path to save results')    
    args = parser.parse_args()

    try:
        with open(args.map_summary,"r") as fin:
            map_summary=json.load(fin)
    except Exception as e:
        print("Failed to open file - exiting")
        sys.exit(-1)
    
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_text=room_likelihood(map_summary, args.tgt_class, include_furniture=True, include_relationships=False)
    map_summary['llama']={'room': {}, 'objects, with room': {}, 'objects, no room': {}}
    map_summary['llama']['room']={'room+furniture': {'input': input_text}}
    map_summary['llama']['room']['room+furniture']['output']=get_llm_response(quantized_model, tokenizer, input_text)
    print(map_summary['llama']['room']['room+furniture']['output'])

    dkey="0.50"
    input_text=object_likelihood(map_summary, args.tgt_class, dkey, include_room=False)
    print(f"************ {dkey} - no room *************")
    if input_text is not None:
        map_summary['llama']['objects, no room'][dkey]={'input': input_text, 'output': get_llm_response(quantized_model, tokenizer, input_text)}
        print(map_summary['llama']['objects, no room'][dkey]['output'])

    input_text=object_likelihood(map_summary, args.tgt_class, dkey, include_room=True)
    print(f"************ {dkey} - with room *************")
    if input_text is not None:
        map_summary['llama']['objects, with room'][dkey]={'input': input_text, 'output': get_llm_response(quantized_model, tokenizer, input_text)}
        print(map_summary['llama']['objects, with room'][dkey]['output'])

    # for dkey in map_summary['object_results'].keys():
    #     input_text=object_likelihood(map_summary, args.tgt_class, dkey)
    #     print(f"************ {dkey} *************")
    #     if input_text is not None:
    #         map_summary['object_results'][dkey]['llama_output']=get_llm_response(quantized_model, tokenizer, input_text)
    #         print(map_summary['object_results'][dkey]['llama_output'])

    with open(args.save_file,"w") as fout:
        json.dump(map_summary, fout)    