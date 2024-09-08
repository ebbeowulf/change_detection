from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
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
        pdb.set_trace()
        # Get the clip probabilities
        outputs = self.process_image(image,threshold)
        # self.set_data(outputs,image.size,threshold)
        
        if save_fileName is not None:
            save_data={'outputs': outputs, 'image_size': image.size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Convert the PIL image to opencv format and return
        return np.array(image) #[:,:,::-1]

    def process_image_tensor(self, image: torch.tensor, threshold=0.5): #assume image is in BGR format as per standard opencv
        #need to optimize this to use GPU
        pil_image=Image.fromarray(image.cpu().numpy()[:,:,::-1])
        return self.process_image(pil_image,threshold=threshold)

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
            P2=torch.sigmoid(outputs.logits).to('cpu').numpy()
        else:
            P2=torch.sigmoid(outputs.logits.unsqueeze(0)).to('cpu').numpy()
        for dim in range(P2.shape[0]):
            self.max_probs[dim]=P2[dim,:,:].max()
            # print("%s = %f"%(self.prompts[dim],self.max_probs[dim]))            
            self.probs[dim]=cv2.resize(P2[dim,:,:],(image_size[0],image_size[1]))
            self.masks[dim]=self.probs[dim]>threshold

        # preds = outputs.logits.unsqueeze(1)
        # P2=self.sigmoid(preds.numpy())
        # for dim in range(preds.shape[0]):
        #     self.max_probs[dim]=P2[dim,0,:,:].max()
        #     print("%s = %f"%(self.prompts[dim],self.max_probs[dim]))            
        #     self.probs[dim]=cv2.resize(P2[dim,0,:,:],(image_size[0],image_size[1]))
        #     self.masks[dim]=self.probs[dim]>threshold
        #     self.build_dbscan_boxes(dim,threshold)
        # else:
        #     pdb.set_trace()
        #     preds = outputs.logits.unsqueeze(0)
        #     P2=self.sigmoid(preds.numpy())
        #     self.probs[0]=cv2.resize(P2[0][0],(image_size[0],image_size[1]))
        #     self.max_probs[0]=P2[0][0].max()
        #     self.masks[0]=self.probs[0]>threshold
        #     self.build_dbscan_boxes(0,threshold)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('tgt_prompt',type=str,default=None,help='specific prompt for clip class')
    parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
    args = parser.parse_args()

    CS=clip_seg([args.tgt_prompt])
    #for i in range(1000):
    #    print(i)
    image=CS.process_file(args.image, threshold=args.threshold)
    mask=CS.get_mask(0)
    if mask is None:
        print("Something went wrong - no mask to display")
    else:
        cv_image=np.array(image).astype(np.uint8)
        #pdb.set_trace()
        IM=cv2.bitwise_and(cv_image,cv_image,mask=mask.astype(np.uint8))
        cv2.imshow("res",IM)
        cv2.waitKey()
    
