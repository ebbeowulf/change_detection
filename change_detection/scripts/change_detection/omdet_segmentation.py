from transformers import AutoProcessor, OmDetTurboForObjectDetection
from ultralytics import SAM
import torch
import cv2
import numpy as np
import argparse
from change_detection.segmentation import image_segmentation
from PIL import Image
import pdb
import pickle

#from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class omdet_segmentation(image_segmentation):
    def __init__(self, prompts):
        print("Reading OmDet-Turbo model")
        self.processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
        self.model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
        self.sam_model = SAM('sam2.1_l.pt')
        if DEVICE==torch.device("cuda"):
            self.model.cuda()
            # Consider moving SAM model to GPU as well if memory allows
            self.sam_model.to(DEVICE)

        self.prompts=prompts
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label }
        self.image_size = None # Store image size when data is set
        self.clear_data()

    def clear_data(self):
        # Clear stored results for a new image
        self.masks = {}
        self.boxes = {}
        self.scores = {}
        self.max_probs = {}
        self.probs = {}
        self.image_size = None

    def sigmoid(self, arr):
        return (1.0/(1.0+np.exp(-arr)))

    def load_file(self, fileName, threshold=0.5):
        try:
            # Otherwise load the file             
            with open(fileName, 'rb') as handle:
                save_data=pickle.load(handle)
                self.clear_data()
                if save_data['prompts']==self.prompts:
                    self.set_data(save_data['outputs'])
                    return True
                else:
                    print("Prompts in saved file do not match ... skipping")
        except Exception as e:
            print(e)
        return False

    def process_file(self, fName, threshold=0.25, save_fileName=None):
        # Predict with the model
        cv_image=cv2.imread(fName,-1)
        results=self.process_image(cv_image, threshold)
        if save_fileName is not None:
            save_data={'outputs': results, 'image_size': self.image_size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cv_image

    def process_image(self, image, threshold=0.25):
        self.clear_data()
        
        # If image is a numpy array, convert to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        self.image_size = image_pil.size
        
        try:
            # Prepare inputs for OmDet
            inputs = self.processor(text=self.prompts, images=image_pil, return_tensors="pt")
            inputs = inputs.to(DEVICE)
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results = self.processor.post_process_object_detection(
                outputs, 
                threshold=threshold,
                target_sizes=[(image_pil.height, image_pil.width)]
            )[0]
            
            # Extract detections
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]
            
            if len(boxes) > 0:
                # Run SAM with OmDet bounding boxes
                sam_results = self.sam_model(image_pil, bboxes=boxes)
                
                # Process each mask
                for i, mask in enumerate(sam_results[0].masks):
                    label_idx = labels[i].item()
                    label_name = self.model.config.id2label[label_idx]
                    
                    # Find which prompt this corresponds to
                    for prompt_idx, prompt in enumerate(self.prompts):
                        if prompt.lower() in label_name.lower() or label_name.lower() in prompt.lower():
                            # Store bounding box
                            if prompt_idx not in self.boxes:
                                self.boxes[prompt_idx] = []
                                self.masks[prompt_idx] = []
                                self.scores[prompt_idx] = []
                                self.max_probs[prompt_idx] = 0.0
                                self.probs[prompt_idx] = torch.zeros(mask.data.shape, device=DEVICE)
                            
                            # Add box and score
                            self.boxes[prompt_idx].append((scores[i].item(), boxes[i].cpu().numpy()))
                            
                            # Process mask
                            curr_mask = mask.data.squeeze()
                            self.masks[prompt_idx].append(curr_mask)
                            
                            # Update probability map
                            prob_array = scores[i].item() * curr_mask
                            self.scores[prompt_idx].append(prob_array)
                            
                            # Update max probability
                            self.max_probs[prompt_idx] = max(self.max_probs[prompt_idx], scores[i].item())
                            
                            # Update combined probability map
                            self.probs[prompt_idx] = torch.maximum(self.probs[prompt_idx], prob_array)
                
                return sam_results
            else:
                print("No objects detected by OmDet.")
                return None
            
        except Exception as e:
            print(f"Exception during OmDet inference: {e}")
            return None

    def process_image_numpy(self, image: np.ndarray, threshold=0.25):
        return self.process_image(image, threshold=threshold)
    
    # The source parameter is already handled in the base class
    # We only need to keep overrides if they have specialized functionality
    # beyond what the base class provides

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('--tgt-class',type=str,default=None,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.25,help='threshold to apply during computation')
    args = parser.parse_args()

    omdet = omdet_segmentation([args.tgt_class])
    img = omdet.process_file(args.image, args.threshold)
    msk = omdet.get_mask(args.tgt_class)
    
    if msk is None or len(msk) == 0:
        print(f"No objects of class {args.tgt_class} detected")
        import sys
        sys.exit(-1)
    else:
        print("Compiling mask image")
        # Combine masks if multiple were detected
        combined_mask = torch.zeros_like(msk[0])
        for mask in msk:
            combined_mask = torch.maximum(combined_mask, mask)
        
        # Convert to numpy for display
        mask_np = combined_mask.cpu().numpy().astype(np.uint8)
        IM = cv2.bitwise_and(img, img, mask=mask_np)

    cv2.imshow("result", IM)
    cv2.waitKey()
