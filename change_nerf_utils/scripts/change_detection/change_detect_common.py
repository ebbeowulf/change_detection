import argparse
from PIL import Image
import numpy as np
import cv2
import pdb

class image_comparator():
    def __init__(self, 
                 detection_targets:list,
                 openVocab_detector:str="CLIPSEG"):

        # self.use_nerf=use_nerf
        self.detection_targets=detection_targets
        if openVocab_detector=="CLIPSEG":
            from clip_segmentation import clip_seg
            self.detect=clip_seg(self.detection_targets)
        elif openVocab_detector=="YOLO_WORLD":
            from yolo_world_segmentation import yolo_world_segmentation
            self.detect=yolo_world_segmentation(self.detection_targets)
        else:
            print(f"Method {openVocab_detector} not currently supported - exiting")
            import os
            os.exit(-1)

        self.detect_id=None

    def openvocab_segmentation(self, 
                      PIL_image1,
                      PIL_image2): 
        self.latest_result=dict()
        for query in self.detection_targets:
            self.latest_result[query]={'prob1': None, 'prob2': None}
        
        self.detect.process_image(PIL_image1)
        for query in self.detection_targets:
            self.latest_result[query]['prob1'] = self.detect.get_prob_array(query)

        self.detect.process_image(PIL_image2)
        for query in self.detection_targets:
            self.latest_result[query]['prob2'] = self.detect.get_prob_array(query)

    def get_change_prob(self,
                        query:str,
                        is_positive_change:bool=True):
        if query in self.latest_result:
            if self.latest_result[query]['prob1'] is not None and self.latest_result[query]['prob2'] is not None:
                if is_positive_change:
                    return self.latest_result[query]['prob2']-self.latest_result[query]['prob1']
                else:
                    return self.latest_result[query]['prob1']-self.latest_result[query]['prob2']
        return None 
    
    
########################################################
## Search terms suggested by the LLM for a living room space
#      General clutter, Small items on surfaces, Floor-level objects, Decorative and functional items, Trash items


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image1',type=str,help='location of before image')
    parser.add_argument('image2',type=str,help='location of the after image')
    parser.add_argument('--queries', type=str, nargs='*', default=["General clutter", "Small items on surfaces", "Floor-level objects", "Decorative and functional items", "Trash items"],
                help='Set of target queries to build point clouds for - default is [General clutter, Small items on surfaces, Floor-level objects, Decorative and functional items, Trash items]')
    parser.add_argument('--threshold',type=float, default=0.5, help="fixed threshold to apply for change detection (default=0.1)")
    args = parser.parse_args()

    prompts = [ s.lower() for s in args.queries ]
    IC=image_comparator(prompts)
    image1=Image.open(args.image1)
    image2=Image.open(args.image2)
    IC.openvocab_segmentation(image1,image2)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(image2)
    image2np=np.array(image2)
    for query_idx, query in enumerate(prompts):
        delta=IC.get_change_prob(query).cpu().numpy()
        print(f"{query} / MAX DELTA={delta.max()}")
        mask=(delta>args.threshold).astype(np.uint8)
        d_mask = (1-mask).astype(np.uint8)
        red_image=np.ones(image2np.shape,dtype=np.uint8)
        red_image[:,:,2]=255
        pos=cv2.bitwise_and(image2np,image2np,mask=d_mask)+cv2.bitwise_and(red_image,red_image,mask=mask)
        plt.figure(query_idx+2)
        plt.imshow(pos)
        plt.title(query)

    plt.show()
