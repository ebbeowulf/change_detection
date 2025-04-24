from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
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

class clip_seg(image_segmentation):
    def __init__(self, prompts):
        print("Reading model")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
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

        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        #threshold = 0.5 # Threshold for post-processing
        #NMS_THRESHOLD = 0.5 # Non-Maximum Suppression threshold

        #print("Running Clip Inference...")
        try:
             # inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding="max_length", return_tensors="pt")
            if len(self.prompts)>1:
                inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding=True, return_tensors="pt")
            else:
                # Adding padding here always throws an error
                inputs = self.processor(text=self.prompts, images=[image], return_tensors="pt")
            inputs.to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
            print(f"Exception during inference step - returning {e}")
            return
        
        # Step 1: Process CLIPSeg logits to get initial masks
        if len(outputs.logits.shape) == 3:  # Check for batch dimension
            P2 = torch.sigmoid(outputs.logits)
        else:
            P2 = torch.sigmoid(outputs.logits.unsqueeze(0))

        # Resize logits to original image size
        P2_large = torch.nn.functional.interpolate(
            P2.unsqueeze(0), size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False
        )[0]  # Shape: (num_prompts, H, W)
        
        # Generate initial masks
        initial_masks = (P2_large > threshold).cpu().numpy().astype(np.uint8)  # Shape: (num_prompts, H, W)
        # Step 2: Check if CLIPSeg detected any objects across all prompts
        clipseg_detected = False
        for dim in range(initial_masks.shape[0]):
            if np.sum(initial_masks[dim]) > 0:  # Check if the mask has any non-zero pixels
                clipseg_detected = True
                break
        class_ids = []
        for idx in range(P2_large.shape[0]):
            class_ids.append(idx)

        #Refine with SAMv2 if objects were detected
        if clipseg_detected:
            sam_results_combined = []  # We'll gather all SAM masks here

            for i, mask in enumerate(initial_masks):
                # Step 1: Skip small/noisy masks
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                # Largest contour only
                cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(cnt) < 100:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Crop image
                cropped_img = image_rgb[y1:y2, x1:x2]

                # Compute centroid and adjust to crop
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                rel_cx = cx - x1
                rel_cy = cy - y1

                # Run SAM on the cropped image
                sam_result = self.sam_model(cropped_img, points=[[rel_cx, rel_cy]])

                #pdb.set_trace()
                # Map SAM mask back to original image size
                full_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                mask_sam = sam_result[0].masks[0].data.squeeze(0)  # shape (H_crop, W_crop)
                full_mask[y1:y2, x1:x2] = mask_sam.cpu().numpy()
                
                sam_result[0].masks = [torch.tensor(full_mask, dtype=torch.uint8, device=DEVICE).unsqueeze(0)]
                
                # Add class ID & confidence
                sam_result[0].class_ids = [class_ids[i]]  
                sam_result[0].probs = P2_large[i] 

                sam_results_combined.append(sam_result)

            if sam_results_combined:
                self.set_data(sam_results_combined)
                return sam_results_combined
            else:
                print("No valid SAM masks generated.")
                return None
        else:
            print("No objects detected by ClipSeg.")
            return None

    def set_data(self, sam_results_combined):
        """Set internal data from SAM results and optional data."""
        self.clear_data()

        if not sam_results_combined:
            print("No valid data to set.")
            return
        #pdb.set_trace()
        for sam_results in sam_results_combined:
            class_ids = np.array(list(sam_results[0].class_ids))   
            confs = sam_results[0].probs.cpu().numpy()  
            if class_ids is not None and confs is not None:
                if sam_results[0].masks and len(sam_results[0].masks) > 0:
                    
                    for i, mask in enumerate(sam_results[0].masks):
                        
                        cls = int(class_ids[i])
                        
                        prob_array = (sam_results[0].probs[i] * mask.data).squeeze()
                        # Store mask and probabilities
                        if cls in self.masks:
                            self.masks[cls] = self.masks[cls] + mask.data.squeeze()
                            self.max_probs[cls] = np.maximum(self.max_probs[cls], confs[i])
                            self.probs[cls] = torch.maximum(self.probs[cls], prob_array)
                        else:
                            self.masks[cls] = mask.data.squeeze()
                            self.max_probs[cls] = confs[i]
                            self.probs[cls] = prob_array
                else:
                    print("No masks returned by SAM.")
            
            else:
                print("No valid class_ids or confs data.")
    

    def process_image_numpy(self, image: np.ndarray, threshold=0.25):
        return self.process_image(image, threshold=threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('--tgt-class',type=str,default=None,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.25,help='threshold to apply during computation')
    args = parser.parse_args()

    CS=clip_seg([args.tgt_class])
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


