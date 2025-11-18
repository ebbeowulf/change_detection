# Query utilities for the LLM server including:
#    single_level_results_template: a class for requesting a predefined results 
#                                       format from the LLM and parsing the output
#    send_data_to_llm: function for communicating with the LLM

import socket
import json
import struct
import io
import numpy as np
from PIL import Image
import cv2
import json
import pdb

HOST = 'localhost'
PORT = 5001
BUFFER_SIZE = 4096

def string_to_bool(input:str):
    low_input=input.lower()
    false=["false","no","wrong","0"]
    true=["true","yes","valid","1"]
    for ff in false:
        if ff in low_input:
            return False
    for tt in true:
        if tt in low_input:
            return True
    return None

#Single level results template assumes
#   that there are no lists or dictionaries in the resulting JSON string
class single_level_results_template():
    def __init__(self, keys:list, types:list, descriptions:list):
        if len(keys)!=len(types) or len(types)!=len(descriptions):
            raise Exception("Results template creation requires same number of keys, types, and descriptions")

        self.template=dict()
        for key_idx, key in enumerate(keys):
            self.template[key]={'type': types[key_idx],'description':descriptions[key_idx]}
            
    def generate_format_prompt(self):
        prompt="{"
        for key_idx, key in enumerate(self.get_keys()):
            if key_idx>0:
                prompt+=", "
            prompt+=f"'{key}': {self.template[key]['description']}"
        prompt+="}"   
        return prompt

    def get_type(self, key, value):
        try:
            if self.template[key]['type']==str:
                return value
            elif self.template[key]['type']==bool:
                if type(value)==bool:
                    return value
                else:
                    return string_to_bool(str(value))
            elif self.template[key]['type']==int:
                return int(value)
            elif self.template[key]['type']==float:
                return float(value)
        except Exception as e:
            return None
        return None
            
    def get_keys(self):
        return self.template.keys()
    
    def recover_json(self, message):
        start_index = message.find('{')
        end_index = message.find('}', start_index + 1)
        json_str=message[start_index:end_index+1].replace('\'','"')
        print(json_str)
        json_out={key:None for key in self.template}
        try:
            res=json.loads(json_str)
            # need to enforce some type constraints at this point
            for key in self.get_keys():
                if key in res:
                    json_out[key]=self.get_type(key, res[key])
            return json_out
        except Exception as e:
            return json_out
    

# Function for communicating with llm server
#    accepts both text and images by default
def send_data_to_llm(text, numpy_images):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        header = {
            "message": text,
            "image_count": len(numpy_images)
        }
        header_bytes = json.dumps(header).encode()
        s.sendall(struct.pack('!I', len(header_bytes)))
        s.sendall(header_bytes)

        for idx, arr in enumerate(numpy_images):
            # Convert NumPy array to PNG bytes
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(arr_rgb.astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            data = buffer.getvalue()

            name = f"image_{idx}.png"
            name_bytes = name.encode()

            s.sendall(struct.pack('!I', len(name_bytes)))
            s.sendall(name_bytes)
            s.sendall(struct.pack('!I', len(data)))
            s.sendall(data)

        response = s.recv(BUFFER_SIZE)
        return json.loads(response.decode())