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
        print("Reading model")
        self.processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
        self.model = OmDetTurboForObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
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
        # Predict with the model
        cv_image=cv2.imread(fName,-1)
        results=self.process_image(cv_image, threshold)
        if save_fileName is not None:
            save_data={'outputs': results, 'image_size': self.image_size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cv_image

    def process_image(self, cv_image, threshold=0.25):
        self.clear_data()
        #pdb.set_trace()     
        #cv_image = np.array(image) # Keep in RGB for SAM

        #threshold = 0.5 # Threshold for post-processing
        #NMS_THRESHOLD = 0.5 # Non-Maximum Suppression threshold

        #print("Running OMDET Inference...")
        try:
            inputs = self.processor(text=self.prompts, images=[cv_image],return_tensors="pt")
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
            target_sizes=torch.tensor([cv_image.size[::-1]]), # Target size (height, width)
            threshold=threshold
        )[0]
        if results and results['boxes'] is not None and len(results['boxes']) > 0:
            # Run SAM with YOLO bounding boxes    
            sam_results = self.sam_model(cv_image, bboxes=results['boxes'])
            sam_results[0].class_ids=results['labels']
            sam_results[0].confs=results['scores']
            # Pass SAM results and YOLO data to set_data       
            self.set_data(sam_results)
            return sam_results
        else:
            print("No objects detected by OmDet.")
            return None

    def set_data(self, sam_results):
        """Set internal data from SAM results and optional YOLO data."""
        self.clear_data()
        class_ids = sam_results[0].class_ids.cpu().numpy()        
        confs = sam_results[0].confs.cpu().numpy()        
        boxes = sam_results[0].boxes.xyxy.cpu().numpy()        
        # Handle case from process_image (SAM results + YOLO data)
        if sam_results and class_ids is not None and confs is not None and boxes is not None:
            if sam_results[0].masks is not None:
                # pdb.set_trace()
                # masks = sam_results[0].masks # .data.cpu().numpy()  # SAM masks as NumPy array
                # if len(masks) != len(class_ids):
                #     raise ValueError("Number of masks must match number of detections")
                
                for i, mask in enumerate(sam_results[0].masks):
                    # Convert data type
                    # if mask.dtype == bool:
                    #     mask = mask.astype(np.uint8)
                    # elif mask.dtype == np.float32:
                    #     mask = (mask * 255).astype(np.uint8)
                    # else:
                    #     print(f"Unexpected mask dtype: {mask.dtype}")
                    #     continue

                    cls = int(class_ids[0])
                    # prob = confs[i]
                    # box = boxes[i]
                    
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
                    prob_array = (sam_results[0].confs[i] * mask.data).squeeze()
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
    parser.add_argument('--tgt-class',type=str,default=None,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.25,help='threshold to apply during computation')
    args = parser.parse_args()

    CS=omdet_segmentation([args.tgt_class])
    img=CS.process_file(args.image,args.threshold)
    msk=CS.get_mask(args.tgt_class)
    if msk is None:
        print("No objects of class %s detectd"%(args.tgt_class))
        sys.exit(-1)
    else:
        print("compiling mask image")                        
        IM=cv2.bitwise_and(img,img,mask=msk.astype(np.uint8))

    cv2.imshow("res",IM)
    cv2.waitKey()


