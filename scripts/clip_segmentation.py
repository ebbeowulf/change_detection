from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import cv2
import numpy as np
import argparse
from segmentation import image_segmentation
from PIL import Image
import pdb

#from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb

class clip_seg(image_segmentation):
    def __init__(self, prompts):
        print("Reading model")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.prompts=prompts
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label }
        self.clear_data()
            
    def sigmoid(self, arr):
        return (1.0/(1.0+np.exp(-arr)))
    
    def process_file(self, fName, threshold=0.2):
        # Need to use PILLOW to load the color image - it has an impact on the clip model???
        image = Image.open(fName)
        # Get the clip probabilities
        self.process_image(image)
        
        # Convert the PIL image to opencv format and return
        return np.array(image)[:,:,::-1]

    def process_image(self, image, threshold=0.2):
        print("Clip Inference")
        self.clear_data()
        try:
            inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding="max_length", return_tensors="pt")
            # predict
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
            print("Exception during inference step - returning")
            return

        if len(self.prompts)>1:
            preds = outputs.logits.unsqueeze(1)
            P2=self.sigmoid(preds.numpy())
            for dim in range(preds.shape[0]):
                self.max_probs[dim]=P2[dim,0,:,:].max()
                print("%s = %f"%(self.prompts[dim],self.max_probs[dim]))            
                self.probs[dim]=cv2.resize(P2[dim,0,:,:],(image.size[0],image.size[1]))
                self.masks[dim]=self.probs[dim]>threshold
        else:
            preds = outputs.logits.unsqueeze(0)
            P2=self.sigmoid(preds.numpy())
            self.probs[0]=cv2.resize(P2[0],(image.size[0],image.size[1]))
            self.max_probs[0]=P2[0].max()
            self.masks[0]=self.probs[0]>threshold
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('tgt_prompt',type=str,default=None,help='specific prompt for clip class')
    parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
    args = parser.parse_args()

    CS=clip_seg([args.tgt_prompt])
    image=CS.process_file(args.image, threshold=args.threshold)
    mask=CS.get_mask(0)
    if mask is None:
        print("Something went wrong - no mask to display")
    else:
        cv_image=np.array(image).astype(np.uint8)
        IM=cv2.bitwise_and(cv_image,cv_image,mask=mask.astype(np.uint8))
        cv2.imshow("res",IM)
        cv2.waitKey()
    