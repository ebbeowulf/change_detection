#Nvidia API Key
from openai import OpenAI
import argparse
import json
import sys
from query_generation import room_likelihood, object_likelihood
import pdb

API_KEY="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"

def get_llm_response(client, input_text):
  completion = client.chat.completions.create(
    model="meta/llama-3.1-405b-instruct",
    messages=[{"role":"user","content":input_text}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
  )

  combined_string=""
  for chunk in completion:
    if chunk.choices[0].delta.content is not None:
      combined_string+=chunk.choices[0].delta.content
  return combined_string

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

    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = API_KEY
    )

    # input_text=room_likelihood(map_summary, args.tgt_class, include_furniture=True, include_relationships=False)
    map_summary['llama']={'room': {}, 'objects, with room': {}, 'objects, no room': {}}
    # map_summary['llama']['room']={'room+furniture': {'input': input_text}}
    # map_summary['llama']['room']['room+furniture']['output']=get_llm_response(client, input_text)
    # print(map_summary['llama']['room']['room+furniture']['output'])

    dkey="0.50"
    input_text=object_likelihood(map_summary, args.tgt_class, dkey, include_room=True)
    print(f"************ {dkey} - with room *************")
    if input_text is not None:
        print(input_text)
        map_summary['llama']['objects, with room'][dkey]={'input': input_text, 'output': get_llm_response(client, input_text)}
        print(map_summary['llama']['objects, with room'][dkey]['output'])

        with open(args.save_file,"w") as fout:
            json.dump(map_summary, fout)    