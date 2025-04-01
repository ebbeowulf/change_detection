from ultralytics import YOLO
import pdb
import cv2
from change_detection.segmentation import image_segmentation
import argparse
import cv2
import numpy as np
import sys
import pickle

class yolo_segmentation(image_segmentation):
    def __init__(self, prompts, model_name='yolov8x-worldv2.pt'):
        # Load a model
        self.model_name=model_name
        self.model=None
        self.prompts=prompts
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label }
        self.loaded_fileName=None # used to track the last file loaded
        self.clear_data()
    
    def set_classes(self, results):
        if self.label2id is None:
            self.id2label = results.names
            self.label2id = { self.id2label[key]: key for key in self.id2label}

    # def clear_data(self):
    #     self.cl_boxes  = np.zeros((0,4),dtype=float)
    #     self.cl_labelID= []
    #     self.cl_probs  = []
    #     super().clear_data()

    # Clear model to save a smaller file
    def clear_model(self):
        self.model=None

    def process_file(self, fName, threshold=0.25, save_fileName=None):
        if self.model is None:
            print("Loading model")
            self.model = YOLO(self.model_name)  # load an official model
            print("Model load finished")

        # Predict with the model
        cv_image=cv2.imread(fName,-1)
        results=self.process_image(cv_image, threshold)
        if save_fileName is not None:
            save_data={'outputs': results, 'image_size': image.size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cv_image

    def load_file(self, fileName, threshold=None):
        try:
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


    def load_prior_results(self, results):
        self.set_data(results)

    def process_image(self, cv_image, threshold=0.25):
        # Predict with the model
        if len(self.prompts)>1:
            yolo_model.set_classes(self.prompts)
            
        yolo_results = self.model(cv_image, conf=threshold)  # predict on an image
        if yolo_results and yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy  # Tensor of bounding boxes
            boxes_list = boxes.cpu().numpy().tolist()  # Convert to a list of lists
            sam_results = sam_model(image_path, bboxes=boxes_list)

        self.set_data(sam_results)
        return sam_results
          
          
    def set_data(self, results):
        self.set_classes(results)
        self.clear_data()
        for result in results:
            cls = int(result.boxes.cls.numpy())
            prob = result.boxes.conf.numpy()[0]
            # Save clusters
            if cls not in self.boxes:
                self.boxes[cls]=[]        
            self.boxes[cls].append((prob, result.boxes.xyxy.numpy()[0]))

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
    
    def plot(self, img, results):
        for result in results:
            for box in result.boxes:
                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                label = results[0].names[int(box.cls)]
                cv2.rectangle(img, (left, top),(right, bottom), (255, 0, 0), 2)

                cv2.putText(img, label,(left, bottom-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Filtered Frame', img)
        cv2.waitKey(0)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('--tgt-class',type=str,default=None,help='specific object class to display')
    parser.add_argument('--threshold',type=float,default=0.25,help='threshold to apply during computation')
    args = parser.parse_args()

    CS=yolo_segmentation()
    img, res=CS.process_file(args.image,args.threshold)
    if args.tgt_class is None:
        IM=CS.plot(img, res)
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
