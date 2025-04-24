import torch
import cv2
import numpy as np
import argparse
from PIL import Image
import pickle
import pdb
from copy import deepcopy
import os
from datetime import datetime

from change_detection.segmentation import image_segmentation
from change_detection.omdet_segmentation import omdet_segmentation
from change_detection.clip_segmentation import clip_seg
from change_detection.yolo_world_segmentation import yolo_world_segmentation
from change_detection.owl_segmentation import owl_segmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hybrid_segmentation(image_segmentation):
    def __init__(self, prompts, classifier='hybrid'):
   
        print("Initializing hybrid segmentation model")
        
        # Define models to use
        self.all_models = ["omdet", "clip"]
        
        # Save initialization parameters
        self.prompts = prompts
        self.id2label = {idx: key for idx, key in enumerate(self.prompts)}
        self.id2label = {}
        idx = 0
        for model in self.all_models:
            for prompt in self.prompts[model]:
                self.id2label[idx] = prompt
                idx += 1

        # Create reverse mapping
        self.label2id = {self.id2label[key]: key for key in self.id2label}
        
        # Initialize individual models if specified
        self.models = {}
        
        if "omdet" in self.use_models:
            print("Initializing OmDet segmentation model")
            self.models["omdet"] = omdet_segmentation(prompts, classifier='hybrid')
            
        if "clip" in self.use_models:
            print("Initializing CLIP segmentation model")
            self.models["clip"] = clip_seg(prompts, classifier='hybrid')
            
        if "yolo_world" in self.use_models:
            print("Initializing YOLO World segmentation model")
            self.models["yolo_world"] = yolo_world_segmentation(prompts, model_name='yolov8x-worldv2.pt')
            
        if "owl" in self.use_models:
            print("Initializing OWL segmentation model")
            self.models["owl"] = owl_segmentation(prompts)
        
        # Initialize data structures for segmentation results
        self.clear_data()
        self.model_results = {}  # Store results from each model
        
    def load_file(self, fileName, threshold=0.5):
        """Load saved segmentation results from a file."""
        try:
            # Load the file             
            with open(fileName, 'rb') as handle:
                save_data = pickle.load(handle)
                self.clear_data()
                if save_data['prompts'] == self.prompts:
                    if 'model_results' in save_data:
                        self.model_results = save_data['model_results']
                    self.set_data(save_data['outputs'])
                    return True
                else:
                    print("Prompts in saved file do not match... skipping")
        except Exception as e:
            print(f"Error loading file: {e}")
        return False

    def process_file(self, fName, threshold=0.25, save_fileName=None, save_individual_models=False):
        # Load and process the image
        cv_image = cv2.imread(fName, -1)
    
        results = self.process_image(cv_image, threshold, save_individual_models)
        
        # Save results if a save file name is provided
        if save_fileName is not None and results is not None:
            save_data = {
                'outputs': results, 
                'model_results': self.model_results,
                'prompts': self.prompts,
                'image_size': cv_image.shape[:2]
            }
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save individual model results if requested
            if save_individual_models and self.model_results:
                base_name = save_fileName.rsplit('.', 1)[0]
                for model_name, model_data in self.model_results.items():
                    if model_name in self.models and hasattr(self.models[model_name], 'masks') and self.models[model_name].masks:
                        model_save_path = f"{base_name}.{model_name}.pkl"
                        model_save_data = {
                            'outputs': self.model_results[model_name].get('outputs', None),
                            'masks': model_data['masks'],
                            'probs': model_data['probs'],
                            'max_probs': model_data['max_probs'],
                            'boxes': model_data['boxes'],
                            'prompts': self.prompts
                        }
                        with open(model_save_path, 'wb') as handle:
                            pickle.dump(model_save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return cv_image

    def process_image(self, cv_image, threshold=0.25):
        self.clear_data()
        
        # Make sure we have initialized models
        if not self.models:
            self._initialize_models()
            
        # Run segmentation with each model
        all_results = {}
        for model_name, model in self.models.items():
            try:
                print(f"Running segmentation with {model_name} model")
                
                # Handle different image format requirements for different models
                if model_name == "owl" or model_name == "omdet":
                    # OWL and OmDet expect PIL format
                    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    results = model.process_image(pil_image, threshold)
                elif model_name == "clip":
                    # CLIP expects OpenCV format and does its own conversion
                    results = model.process_image(cv_image, threshold
                                                  )
                else:
                    # Other models use OpenCV image directly
                    results = model.process_image(cv_image, threshold)
                
                if results is not None:
                    all_results[model_name] = results

            except Exception as e:
                print(f"Error processing with {model_name} model: {e}")
                import traceback
                traceback.print_exc()
        
        # Combine results from all models
        if all_results:
            combined_results = self.combine_results(all_results)
            return combined_results
        else:
            print("No model produced valid results")
            return None
    
    def combine_results(self, all_results):
        """Combine results from all models into a single output."""
        combined_masks = []
        combined_probs = []
        combined_boxes = []
        
        for model_name, results in all_results.items():
            if hasattr(results, 'masks'):
                combined_masks.append(results.masks)
            if hasattr(results, 'probs'):
                combined_probs.append(results.probs)
            if hasattr(results, 'boxes'):
                combined_boxes.append(results.boxes)
        
        # Combine masks and probabilities
        if combined_masks:
            self.masks = torch.cat(combined_masks, dim=0)
        if combined_probs:
            self.probs = torch.cat(combined_probs, dim=0)
        if combined_boxes:
            self.boxes = torch.cat(combined_boxes, dim=0)

        return {
            'masks': self.masks,
            'probs': self.probs,
            'boxes': self.boxes
        }
    
    def set_data(self, combined_results):
        """Set internal data from combined results."""
        self.clear_data()
        
        if combined_results and 'masks' in combined_results:
            self.masks = combined_results['masks']
            self.probs = combined_results['probs']
            self.boxes = combined_results['boxes']
        else:
            print("No valid data to set.")

    def process_image_numpy(self, image: np.ndarray, threshold=0.25):        
        return self.process_image(image, threshold=threshold)

    def _initialize_models(self):
        """Initialize individual models based on use_models list."""
        for model_name in self.use_models:
            if model_name == "omdet":
                self.models["omdet"] = omdet_segmentation(self.prompts)
            elif model_name == "clip":
                self.models["clip"] = clip_seg(self.prompts)
            elif model_name == "yolo_world":
                self.models["yolo_world"] = yolo_world_segmentation(self.prompts)
            elif model_name == "owl":
                self.models["owl"] = owl_segmentation(self.prompts)
        
        # Initialize result containers
        self.model_results = {model: {} for model in self.use_models}
        self.temporal_tracking = {}
        self.detection_history = {}
        self.frame_count = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='location of image to process')
    parser.add_argument('--tgt-class', type=str, default=None, help='specific object class to display')
    parser.add_argument('--threshold', type=float, default=0.25, help='threshold to apply during computation')
    parser.add_argument('--models', type=str, default='omdet,clip,yolo_world,owl', help='comma-separated list of models to use')
    parser.add_argument('--save-individual', action='store_true', help='save individual model results separately')
    parser.add_argument('--save-path', type=str, default=None, help='path to save results')
    args = parser.parse_args()

    # Parse models to use
    use_models = [model.strip() for model in args.models.split(',')]
    
    # Initialize hybrid segmentation model
    if args.tgt_class:
        hybrid_seg = hybrid_segmentation([args.tgt_class], use_models=use_models)
    else:
        # Default COCO classes if no target class is specified
        coco_classes = ['person', 'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'oven', 'sink', 'refrigerator', 'book']
        hybrid_seg = hybrid_segmentation(coco_classes, use_models=use_models)

    # Process the image
    img = hybrid_seg.process_file(
        args.image, 
        args.threshold, 
        args.save_path, 
        args.save_individual,
        not args.disable_temporal,
        args.min_detections,
        args.confidence_boost
    )
    
    if args.tgt_class:
        # Display segmentation for target class
        msk = hybrid_seg.get_mask(args.tgt_class)
        if msk is None:
            print(f"No objects of class {args.tgt_class} detected")
            exit(-1)
        else:
            print("Compiling mask image")
            if isinstance(msk, torch.Tensor):
                msk = msk.cpu().numpy()
            IM = cv2.bitwise_and(img, img, mask=msk.astype(np.uint8))
    else:
        # Display confidence map for all classes
        conf_map = hybrid_seg.get_confidence_map()
        if conf_map is None:
            print("No objects detected")
            exit(-1)
        
        # Normalize for visualization
        conf_map = (conf_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(conf_map, cv2.COLORMAP_JET)
        IM = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

    # Show the result
    cv2.imshow("Hybrid Segmentation Result", IM)
    cv2.waitKey(0)
    cv2.destroyAllWindows()