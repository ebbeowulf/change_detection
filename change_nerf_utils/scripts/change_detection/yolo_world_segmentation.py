from ultralytics import YOLO, SAM
import pdb
import cv2
from change_detection.segmentation import image_segmentation
import argparse
import cv2
import numpy as np
import sys
import pickle

class yolo_world_segmentation(image_segmentation):
    def __init__(self, prompts, model_name='yolov8x-worldv2.pt'):
        # Load a model
        self.model_name=model_name
        self.yolo_model=None
        self.sam_model=None
        self.prompts=prompts
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label }
        self.loaded_fileName=None # used to track the last file loaded
        self.clear_data()
        self.init_model()
    
    def set_classes(self, results):
        if self.label2id is None:
            self.id2label = results.names
            self.label2id = { self.id2label[key]: key for key in self.id2label}

    # Clear model to save a smaller file
    def clear_model(self):
        self.yolo_model=None
        self.sam_model=None

    def init_model(self):
        if self.yolo_model is None:
            print("Loading yolo model")
            self.yolo_model = YOLO(self.model_name)  # load an official model
            print("Yolo Model load finished")
        if self.sam_model is None:
            print("Loading SAM model")
            self.sam_model = SAM('sam2.1_l.pt')  # load an official model
            print("SAM Model load finished")

    def process_file(self, fName, threshold=0.25, save_fileName=None):
        # Predict with the model
        cv_image=cv2.imread(fName,-1)
        results=self.process_image(cv_image, threshold)
        if save_fileName is not None:
            save_data={'outputs': results, 'image_size': self.image_size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cv_image

    def load_file(self, fileName, threshold=None):
        try:
            if self.loaded_fileName == fileName:
                return True
            with open(fileName, 'rb') as handle:
                results = pickle.load(handle)
                if results['outputs'] is not None:
                    self.load_prior_results(results)
                else:
                    print("No Detections found in file")
            self.loaded_fileName = fileName
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            self.loaded_fileName = None
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

    def process_image_numpy(self, image: np.ndarray, threshold=0.25):
        return self.process_image(image, threshold=threshold)
    
    def process_image(self, cv_image, threshold=0.25):
        # Predict with YOLO
        self.yolo_model.set_classes(self.prompts)
        self.image_size = (cv_image.shape[1], cv_image.shape[0])  # (width, height)
        
        yolo_results = self.yolo_model(cv_image, conf=threshold)  # Predict on an image
        if yolo_results and yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
            # Run SAM with YOLO bounding boxes
            sam_results = self.sam_model(cv_image, bboxes=yolo_results[0].boxes.xyxy)
            sam_results[0].class_ids=yolo_results[0].boxes.cls
            sam_results[0].confs=yolo_results[0].boxes.conf
            # Pass SAM results and YOLO data to set_data
            self.set_data(sam_results)
            return sam_results
        else:
            print("No objects detected by YOLO.")
            return None
          

    def sigmoid(self, arr):
        return (1.0/(1.0+np.exp(-arr)))


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
                        self.masks[cls] += mask.data.squeeze()
                        self.max_probs[cls] = max(self.max_probs[cls], confs[i])
                        self.probs[cls] = np.maximum(self.probs[cls], prob_array)
                    else:
                        self.masks[cls] = mask.data.squeeze()
                        self.max_probs[cls] = confs[i]
                        self.probs[cls] = prob_array
            else:
                print("No masks returned by SAM.")
        
        else:
            print("No valid data to set.")
    
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

    CS=yolo_world_segmentation([args.tgt_class])
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
