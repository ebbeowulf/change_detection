from ultralytics import YOLO
import pdb
import cv2
from segmentation import image_segmentation
import argparse
import cv2
import numpy as np
import sys

class yolo_segmentation(image_segmentation):
    def __init__(self, model_name='yolov8n-seg.pt'):
        # Load a model
        self.model_name=model_name
        self.model=None
        self.label2id = None
        self.id2label = None
        self.clear_data()
    
    def set_classes(self, results):
        if self.label2id is None:
            self.id2label = results.names
            self.label2id = { self.id2label[key]: key for key in self.id2label}

    def clear_data(self):
        self.cl_boxes  = np.zeros((0,4),dtype=float)
        self.cl_labelID= []
        self.cl_probs  = []
        super().clear_data()

    # Clear model to save a smaller file
    def clear_model(self):
        self.model=None

    def process_file(self, fName, threshold=0.25):
        if self.model is None:
            print("Loading model")
            self.model = YOLO(self.model_name)  # load an official model
            print("Model load finished")

        # Predict with the model
        cv_image=cv2.imread(fName,-1)
        results=self.process_image(cv_image, threshold)
        return cv_image,results

    def load_prior_results(self, results):
        self.set_data(results)

    def process_image(self, cv_image, threshold=0.25):
        # Predict with the model
        results = self.model(cv_image, conf=threshold)[0].cpu()  # predict on an image
        self.set_data(results)
        return results
    
    def set_data(self, results):
        self.set_classes(results)
        self.clear_data()
        for result in results:
            cls = int(result.boxes.cls.numpy())
            prob = result.boxes.conf.numpy()[0]
            # Save clusters
            self.cl_labelID.append(cls)
            self.cl_probs.append(prob)
            self.cl_boxes = np.vstack((self.cl_boxes, result.boxes.xyxy.numpy()))

            msk_resized=cv2.resize(result.masks.data.numpy().squeeze(),(result.orig_shape[1],result.orig_shape[0]))
            prob_array=prob*msk_resized
            # Save mask + max prob
            if cls in self.masks:
                self.masks[cls] += msk_resized
                self.max_probs[cls] = max(self.max_probs[cls],prob)
                self.probs[cls] = np.maximum.reduce([self.probs[cls], prob_array])
            else:
                self.masks[cls] = msk_resized
                self.max_probs[cls]=prob
                self.probs[cls]=prob_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('--tgt-class',type=str,default=None,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.25,help='threshold to apply during computation')
    args = parser.parse_args()

    CS=yolo_segmentation()
    res=CS.process_file(args.image,args.threshold)
    if args.tgt_class is None:
        IM=res.plot()
    else:
        msk=CS.get_mask(args.tgt_class)
        if msk is None:
            print("No objects of class %s detectd"%(args.tgt_class))
            sys.exit(-1)
        else:
            print("compiling mask image")                        
            IM=cv2.bitwise_and(res.orig_img,res.orig_img,mask=msk.astype(np.uint8))

    cv2.imshow("res",IM)
    cv2.waitKey()