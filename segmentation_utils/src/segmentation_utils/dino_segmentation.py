#################################################################################
##  Code written by Hemanth Indurthi Venkata for his MS Thesis
##      based on the code we wrote together for clip-seg and yolo-world
#################################################################################

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline
from ultralytics import SAM
import torch
import cv2
import numpy as np
import argparse
from segmentation_utils.segmentation import image_segmentation
from PIL import Image
import pickle
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dino_segmentation(image_segmentation):
    def __init__(self, prompts):
        print("Reading model")
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
        self.sam_model = SAM('sam2.1_l.pt')
        if DEVICE==torch.device("cuda"):
            self.model.cuda()
            self.sam_model.to(torch.device("cuda"))

        self.prompts=prompts
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label}
        self.clear_data()

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

    def load_prior_results(self, results):
        """Load prior results from a saved file."""
        # Ensure results is a dictionary from the pickled file
        if isinstance(results, dict) and 'outputs' in results:
            self.image_size = results['image_size']
            self.prompts = results['prompts']
            self.set_data(results['outputs'])
        else:
            raise ValueError("Loaded results must be a dictionary with 'outputs', 'image_size', and 'prompts'")

    def process_file(self, fName, threshold=0.25, save_fileName=None):
        # Need to use PILLOW to load the color image - it has an impact on the clip model???
        image = Image.open(fName)
        outputs = self.process_image(image,threshold)
        
        if save_fileName is not None:
            save_data={'outputs': outputs, 'image_size': image.size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Convert the PIL image to opencv format and return
        return np.array(image)

    def process_image(self, pil_image, threshold=0.25):
        self.clear_data()

        #print("Running DINO Inference...")
        try:
            inputs = self.processor(text=self.prompts, images=pil_image,return_tensors="pt")
            inputs.to(DEVICE)
            # predict
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
            print(f"Exception during inference step - returning {e}")
            return
        # Use processor's post-processing
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            text_labels=self.prompts,
            target_sizes=torch.tensor([pil_image.size[::-1]]), # Target size (height, width)
            threshold=threshold
        )[0]
        if results and results['boxes'] is not None and len(results['boxes']) > 0:
            # Run SAM with YOLO bounding boxes    
            sam_results = self.sam_model(pil_image, bboxes=results['boxes'])
            # Pass SAM results and YOLO data to set_data       
            self.set_data(sam_results)
            return sam_results
        else:
            print("No objects detected by Dino.")
            return None

    def set_data(self, sam_results):
        """Set internal data from SAM results and optional YOLO data."""
        self.clear_data()
        class_ids = sam_results[0].names        
        confs = sam_results[0].boxes.conf.cpu().numpy()        
        boxes = sam_results[0].boxes.xyxy.cpu().numpy()        
        # Handle case from process_image (SAM results + YOLO data)
        if sam_results and class_ids is not None and confs is not None and boxes is not None:
            if sam_results[0].masks is not None:                
                for i, mask in enumerate(sam_results[0].masks):
                    # Convert data type
                    cls = int(class_ids[0])
                    
                    # Store bounding box and confidence
                    if cls not in self.boxes:
                        self.boxes[cls] = []
                    self.boxes[cls].append((confs[i], boxes[i]))
                    
                    # Resize mask to original image size

                    # mask_resized = cv2.resize(
                    #     mask, 
                    #     (self.image_size[0], self.image_size[1]),  # (width, height)
                    #     interpolation=cv2.INTER_NEAREST
                    # )
                    prob_array = (sam_results[0].boxes.conf[i] * mask.data).squeeze()
                    # Store mask and probabilities
                    if cls in self.masks:
                        self.masks[cls] = self.masks[cls] + mask.data.squeeze()
                        self.max_probs[cls] = max(self.max_probs[cls], confs[i])
                        self.probs[cls] = torch.maximum(self.probs[cls], prob_array)
                    else:
                        self.masks[cls] = mask.data.squeeze()
                        self.max_probs[cls] = confs[i]
                        self.probs[cls] = prob_array
            else:
                print("No masks returned by SAM.")
        
        else:
            print("No valid data to set.")
    
    def process_image_numpy(self, image: np.ndarray, threshold=0.25):
        image_pil=Image.fromarray(image)
        return self.process_image(image_pil, threshold=threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('tgt_class',type=str,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.25,help='threshold to apply during computation')
    args = parser.parse_args()

    CS=dino_segmentation([args.tgt_class])
    from PIL import Image
    pil_image=Image.open(args.image)
    # img=CS.process_file(args.image,args.threshold)
    CS.process_image(pil_image,args.threshold)
    msk=CS.get_mask(args.tgt_class)
    prob1 = CS.get_prob_array(0).to('cpu').numpy()
    print(f"Max probability = {prob1.max()}")
    if msk is None:
        print("No objects of class %s detectd"%(args.tgt_class))
        sys.exit(-1)
    else:
        print("compiling mask image")                        
        img=np.array(pil_image)        
        IM=cv2.bitwise_and(img,img,mask=msk.cpu().numpy().astype(np.uint8))

    cv2.imshow("res",IM)
    cv2.waitKey()


