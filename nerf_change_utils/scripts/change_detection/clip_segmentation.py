from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import cv2
import numpy as np
import argparse
from .segmentation import image_segmentation
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
        if DEVICE==torch.device("cuda"):
            self.model.cuda()
        self.prompts=prompts
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label }
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
                    self.set_data(save_data['outputs'],save_data['image_size'],threshold)
                    return True
                else:
                    print("Prompts in saved file do not match ... skipping")
        except Exception as e:
            print(e)
        return False
        
    def process_file(self, fName, threshold=0.5, save_fileName=None):
        # Need to use PILLOW to load the color image - it has an impact on the clip model???
        image = Image.open(fName)
        # Get the clip probabilities
        outputs = self.process_image(image,threshold)
        # self.set_data(outputs,image.size,threshold)
        
        if save_fileName is not None:
            save_data={'outputs': outputs, 'image_size': image.size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Convert the PIL image to opencv format and return
        return np.array(image)

    def process_image_numpy(self, image: np.ndarray, threshold=0.5):
        image_pil=Image.fromarray(image)
        return self.process_image(image_pil, threshold=threshold)
            
    def process_image(self, image: Image, threshold=0.5): #image should be in PIL format
        # print("Clip Inference")
        self.clear_data()
        try:
            # inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding="max_length", return_tensors="pt")
            if len(self.prompts)>1:
                inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding=True, return_tensors="pt")
            else:
                # Adding padding here always throws an error
                inputs = self.processor(text=self.prompts, images=[image], return_tensors="pt")
            inputs.to(DEVICE)
            # predict
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
            print("Exception during inference step - returning")
            return

        self.set_data(outputs,image.size,threshold)
        return outputs
          
    def set_data(self, outputs, image_size, threshold=0.2):
        if len(outputs.logits.shape)==3:  # need to check because of old libraries auto compressing the first dimension
            # P2=torch.sigmoid(outputs.logits).to('cpu').numpy()
            P2=torch.sigmoid(outputs.logits)
        else:
            # P2=torch.sigmoid(outputs.logits.unsqueeze(0)).to('cpu').numpy()
            P2=torch.sigmoid(outputs.logits.unsqueeze(0))
        P2_large=torch.nn.functional.interpolate(P2.unsqueeze(0), size=(image_size[1],image_size[0]), mode='bilinear', align_corners=False)[0]
        for dim in range(P2_large.shape[0]):
            self.max_probs[dim]=float(P2_large[dim].max().cpu())
            # print("%s = %f"%(self.prompts[dim],self.max_probs[dim]))            
            self.probs[dim]=P2_large[dim]
            self.masks[dim]=self.probs[dim]>threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
    parser.add_argument('--options', type=str, default=None, help="Other options: PIL = open with PIL library, CV2 = open with opencv library, CV2_ROTATE = open with opencv and rotate during clip processing")
    args = parser.parse_args()

    CS=clip_seg(args.targets)

    if args.options is None or args.options=="PIL":
        image=CS.process_file(args.image, threshold=args.threshold)
        mask=CS.get_mask(0).cpu().numpy()
    elif args.options =="CV2":
        image=cv2.imread(args.image)        
        CS.process_image_numpy(image, threshold=args.threshold)
        mask=CS.get_mask(0).cpu().numpy()
    elif args.options=="CV2_ROTATE":
        image=cv2.imread(args.image)
        image_rot=np.rot90(image, k=1, axes=(1,0))
        image_pil=Image.fromarray(image_rot)
        CS.process_image(image_pil, threshold=args.threshold)
        mask=np.rot90(CS.get_mask(0),axes=(0,1)).cpu().numpy()
        pdb.set_trace()
    else:
        print(f"Invalid option {args.options}")
        import sys
        sys.exit(-1)

    if mask is None:
        print("Something went wrong - no mask to display")
    else:
        cv_image=np.array(image).astype(np.uint8)
        #pdb.set_trace()
        IM=cv2.bitwise_and(cv_image,cv_image,mask=mask.astype(np.uint8))
        cv2.imshow("res",IM)
        cv2.waitKey()
    
