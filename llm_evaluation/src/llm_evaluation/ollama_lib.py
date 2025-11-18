import sys
import ollama
from vlm import visual_language_model
import os
from PIL import Image
TMP_STORAGE="/tmp/ollama_images/"
valid_models = ["llava:13b", "llava-llama3", "llama4:scout", "moondream"]

class ollama_lib(visual_language_model):
    def __init__(self, model_name: str  = "llama4:scout"):
        if model_name not in valid_models:
            print(f"Error: Invalid model name. Choose from: {', '.join(valid_models)}")
            sys.exit(1)

        os.makedirs(TMP_STORAGE, exist_ok=True)

        self.model_name=model_name
        ollama.pull(model_name)

    def process_input(self, text_message:str, images:list):
        # Ollama requires that we save any images to disk first
        #    Note that we are assuming incoming images are in PIL format, not numpy
        image_paths=[]                        
        for idx, pil_image in enumerate(images):            
            img_path=TMP_STORAGE+f"image_{idx}.png"
            pil_image.save(img_path)
            image_paths.append(img_path)

        return self.run_multi_image_inference(text_message, image_paths)

    def run_multi_image_inference(self, prompt: str, image_path_list: list, print_output: bool=False):
        stream = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt, "images": image_path_list}],
            stream=True,
        )
        message=""
        for chunk in stream:
            message+=chunk["message"]["content"]
            if print_output:
                print(chunk["message"]["content"], end="", flush=True)
        
        return message


