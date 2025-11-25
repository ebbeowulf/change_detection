import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import pickle
import numpy as np
from segmentation_utils.segmentation import image_segmentation

# Based off the example usage files from the SAM3 GitHub Repo:
#           https://github.com/facebookresearch/sam3.git

class sam3_segmentation(image_segmentation):
    def __init__(self, prompts, threshold=0.01):
        # turn on tfloat32 for Ampere GPUs
        # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # use bfloat16 for the entire notebook. If your card doesn't support it, try float16 instead
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # inference mode for the whole notebook. Disable if you need gradients
        torch.inference_mode().__enter__()

        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model, confidence_threshold=threshold)

        self.prompts = prompts
        self.default_threshold=threshold
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label }
        self.clear_data()

    def process_file(self, fName, threshold=0.5, save_fileName=None):
        # Need to use PILLOW to load the color image 
        image = Image.open(fName)

        # Get the clip probabilities
        outputs = self.process_image(image,threshold)
        
        if save_fileName is not None:
            save_data={'outputs': outputs, 'image_size': image.size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Convert the PIL image to opencv format and return
        return np.array(image)
    
    def process_image(self, image: Image, threshold=0.1): #image should be in PIL format
        # print("Clip Inference")
        self.clear_data()
        outputs=dict()
        try:
            inference_state = self.processor.set_image(image)
            for text in self.prompts:
                outputs[self.label2id[text]] = self.processor.set_text_prompt(state=inference_state, prompt=text)
        except Exception as e:
            print("Exception during inference step - returning")
            return

        self.set_data(outputs,threshold)
        return outputs

    def set_box(self, cls, box, output_mask, score):
        boxI=box.floor().cpu().numpy().astype('int')
        self.boxes[cls].append((score,box))        
        self.probs[cls][boxI[1]:boxI[3],boxI[0]:boxI[2]]=self.probs[cls][boxI[1]:boxI[3],boxI[0]:boxI[2]].max(score*output_mask[0,boxI[1]:boxI[3],boxI[0]:boxI[2]])

    def set_data(self, outputs, threshold=0.2):
        for cls in outputs.keys():  
            filter=(outputs[cls]['scores']>threshold).cpu() # which boxes exceed threshold?
            whichV=np.where(filter.numpy())[0]
            self.masks[cls]=outputs[cls]['masks'][whichV].sum(0).squeeze()>0
            self.probs[cls]=torch.zeros(self.masks[cls].shape).cuda()
            self.max_probs[cls]=outputs[cls]['scores'].max().cpu().tolist()
            self.boxes[cls]=[]
            for val in whichV:
                self.set_box(cls, outputs[cls]['boxes'][val], outputs[cls]['masks'][val], outputs[cls]['scores'][val])

if __name__ == '__main__':
    import argparse
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('tgt_class',type=str,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
    parser.add_argument('--options', type=str, default=None, help="Other options: CV2_DRAW = display combined mask using CV2 instead of Meta draw utility")
    args = parser.parse_args()

    CS=sam3_segmentation([args.tgt_class],args.threshold)
    image = Image.open(args.image)
    outputs=CS.process_image(image, threshold=args.threshold)
    if args.options is None:
        from sam3.visualization_utils import plot_results
        import matplotlib.pyplot as plt
        plot_results(image, outputs[0])
        plt.show()
    elif args.options=="CV2_DRAW":
        mask=CS.get_mask(0).cpu().numpy()
        cv_image=np.array(image).astype(np.uint8)[:,:,[2,1,0]]
        IM=cv2.bitwise_and(cv_image,cv_image,mask=mask.astype(np.uint8))
        cv2.imshow("res",IM)
        cv2.waitKey()