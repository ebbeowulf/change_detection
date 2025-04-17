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
from change_detection.clip_sam_segmentation import clip_seg
from change_detection.yolo_world_segmentation import yolo_world_segmentation
from change_detection.owl_segmentation import owl_segmentation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hybrid_segmentation(image_segmentation):
    def __init__(self, prompts, use_models=None, weights=None, yolo_model_name='yolov8x-worldv2.pt'):
        """
        Initialize a hybrid segmentation model that combines multiple models.
        
        Args:
            prompts (list): List of text prompts for segmentation models
            use_models (list): List of model names to use ["omdet", "clip", "yolo_world", "owl"]. Default is all.
            weights (dict): Weights for each model's contribution. Default is equal weights.
            yolo_model_name (str): YOLO model variant to use
        """
        print("Initializing hybrid segmentation model")
        
        # Save initialization parameters
        self.prompts = prompts
        self.id2label = {idx: key for idx, key in enumerate(self.prompts)}
        self.label2id = {self.id2label[key]: key for key in self.id2label}
        
        # Define models to use
        self.all_models = ["omdet", "clip"]
        self.use_models = use_models if use_models else self.all_models
        
        # Define model weights - prioritize YOLO World and OWL for better object detection
        if weights is None:
            self.weights = {model: 1.0/len(self.use_models) for model in self.use_models}
        else:
            self.weights = weights
        
        # Initialize individual models if specified
        self.models = {}
        
        if "omdet" in self.use_models:
            print("Initializing OmDet segmentation model")
            self.models["omdet"] = omdet_segmentation(prompts)
            
        if "clip" in self.use_models:
            print("Initializing CLIP segmentation model")
            self.models["clip"] = clip_seg(prompts)
            
        if "yolo_world" in self.use_models:
            print("Initializing YOLO World segmentation model")
            self.models["yolo_world"] = yolo_world_segmentation(prompts, model_name=yolo_model_name)
            
        if "owl" in self.use_models:
            print("Initializing OWL segmentation model")
            self.models["owl"] = owl_segmentation(prompts)
        
        # Initialize data structures for segmentation results
        self.clear_data()
        self.model_results = {}  # Store results from each model
        
        # Initialize temporal consistency tracking
        self.temporal_tracking = {}  # Track detections across frames
        self.detection_history = {}  # Store detection history
        self.frame_count = 0  # Count processed frames
        
    def clear_data(self):
        """Clear all segmentation data from the model."""
        super().clear_data()
        self.model_results = {}
        
    def clear_temporal_data(self):
        """Clear temporal tracking data."""
        self.temporal_tracking = {}
        self.detection_history = {}
        self.frame_count = 0

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

    def process_file(self, fName, threshold=0.25, save_fileName=None, save_individual_models=False, 
                  temporal_consistency=True, min_detections=2, confidence_boost=0.25):
        """
        Process an image file and perform segmentation.
        
        Args:
            fName: Path to the image file
            threshold: Confidence threshold for segmentation
            save_fileName: Path to save combined results (if None, don't save)
            save_individual_models: Whether to save individual model results separately
            temporal_consistency: Whether to apply temporal consistency
            min_detections: Minimum number of detections required to consider an object valid
            confidence_boost: Amount to boost confidence for temporally consistent detections
        """
        # Load and process the image
        cv_image = cv2.imread(fName, -1)
        
        # Process with temporal consistency if enabled
        if temporal_consistency:
            results = self.process_image_with_temporal_consistency(
                cv_image, threshold, temporal_consistency, min_detections, confidence_boost
            )
        else:
            results = self.process_image(cv_image, threshold, save_individual_models)
        
        # Save results if a save file name is provided
        if save_fileName is not None and results is not None:
            save_data = {
                'outputs': results, 
                'model_results': self.model_results,
                'prompts': self.prompts,
                'temporal_data': {
                    'frame_count': self.frame_count,
                    'detection_history': self.detection_history
                } if temporal_consistency else None
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

    def process_image(self, cv_image, threshold=0.25, save_individual_outputs=False):
        """
        Process an image using multiple segmentation models and combine their results.
        
        Args:
            cv_image: OpenCV image to process
            threshold: Confidence threshold for segmentation
            save_individual_outputs: Whether to save outputs from each model
            
        Returns:
            Combined segmentation results
        """
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
                    results = model.process_image(cv_image, threshold)
                else:
                    # Other models use OpenCV image directly
                    results = model.process_image(cv_image, threshold)
                
                if results is not None:
                    all_results[model_name] = results
                    # Store individual model's processed data
                    self.model_results[model_name] = {
                        'masks': deepcopy(model.masks),
                        'probs': deepcopy(model.probs),
                        'max_probs': deepcopy(model.max_probs),
                        'boxes': deepcopy(model.boxes),
                        'outputs': results if save_individual_outputs else None
                    }
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
        """
        Combine segmentation results from multiple models.
        
        Args:
            all_results: Dictionary of results from each model
            
        Returns:
            Combined segmentation results
        """
        # For each class in prompts
        for cls_id, cls_name in self.id2label.items():
            mask_combined = None
            prob_combined = None
            max_prob = 0.0
            boxes_combined = []
            
            # Combine masks and probabilities from each model
            for model_name, model_data in self.model_results.items():
                weight = self.weights.get(model_name, 1.0 / len(self.model_results))
                
                # Add masks if available for this class
                if cls_id in model_data['masks']:
                    if mask_combined is None:
                        # Initial assignment - handle tensor or numpy array
                        mask_combined = weight * model_data['masks'][cls_id]
                    else:
                        # Combining masks - handle different data types
                        if isinstance(mask_combined, torch.Tensor) and isinstance(model_data['masks'][cls_id], torch.Tensor):
                            # Both are tensors
                            mask_combined = torch.maximum(mask_combined, weight * model_data['masks'][cls_id])
                        else:
                            # Handle mixed types or both numpy arrays
                            # Convert to numpy if needed
                            if isinstance(mask_combined, torch.Tensor):
                                mask_combined = mask_combined.cpu().numpy()
                            mask_to_add = model_data['masks'][cls_id]
                            if isinstance(mask_to_add, torch.Tensor):
                                mask_to_add = mask_to_add.cpu().numpy()
                            mask_combined = np.maximum(mask_combined, weight * mask_to_add)
                
                # Add probability arrays if available
                if cls_id in model_data['probs']:
                    if prob_combined is None:
                        # Initial assignment - handle tensor or numpy array
                        prob_combined = weight * model_data['probs'][cls_id]
                    else:
                        # Combining probabilities - handle different data types
                        if isinstance(prob_combined, torch.Tensor) and isinstance(model_data['probs'][cls_id], torch.Tensor):
                            # Both are tensors
                            prob_combined = torch.maximum(prob_combined, weight * model_data['probs'][cls_id])
                        else:
                            # Handle mixed types or both numpy arrays
                            # Convert to numpy if needed
                            if isinstance(prob_combined, torch.Tensor):
                                prob_combined = prob_combined.cpu().numpy()
                            prob_to_add = model_data['probs'][cls_id]
                            if isinstance(prob_to_add, torch.Tensor):
                                prob_to_add = prob_to_add.cpu().numpy()
                            prob_combined = np.maximum(prob_combined, weight * prob_to_add)
                
                # Update max probability
                if cls_id in model_data['max_probs']:
                    if isinstance(model_data['max_probs'][cls_id], (np.ndarray, torch.Tensor)):
                        # If it's an array or tensor, take the maximum value from it
                        if isinstance(model_data['max_probs'][cls_id], torch.Tensor):
                            model_max_prob = model_data['max_probs'][cls_id].max().item()
                        else:
                            model_max_prob = model_data['max_probs'][cls_id].max()
                        max_prob = max(max_prob, weight * model_max_prob)
                    else:
                        # If it's a scalar, use it directly
                        max_prob = max(max_prob, weight * model_data['max_probs'][cls_id])
                
                # Combine bounding boxes
                if cls_id in model_data['boxes']:
                    boxes_combined.extend(model_data['boxes'][cls_id])
            
            # Store combined results
            if mask_combined is not None:
                self.masks[cls_id] = mask_combined
                self.probs[cls_id] = prob_combined
                self.max_probs[cls_id] = max_prob
                
                # For boxes, we need to perform non-maximum suppression
                if boxes_combined:
                    self.boxes[cls_id] = self.non_maximum_suppression(boxes_combined)
        
        # Return the first model's results format but with our combined data
        # This maintains compatibility with the clustering system
        if all_results and next(iter(all_results.values())) is not None:
            first_result = next(iter(all_results.values()))
            return first_result
            
    def non_maximum_suppression(self, boxes, iou_threshold=0.5):
        """
        Perform non-maximum suppression on bounding boxes.
        
        Args:
            boxes: List of (confidence, box) tuples
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of (confidence, box) tuples
        """
        # Sort boxes by confidence (descending)
        boxes.sort(key=lambda x: x[0], reverse=True)
        
        filtered_boxes = []
        for i, (conf_i, box_i) in enumerate(boxes):
            keep = True
            for conf_j, box_j in filtered_boxes:
                # Calculate IoU
                iou = self.calculate_iou(box_i, box_j)
                if iou > iou_threshold:
                    keep = False
                    break
            
            if keep:
                filtered_boxes.append((conf_i, box_i))
                
        return filtered_boxes
    
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0  # No intersection
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate IoU
        return area_i / (area1 + area2 - area_i)
    
    def get_confidence_map(self):
        """
        Generate a confidence map visualization across all classes.
        
        Returns:
            Visualization of confidence across all classes
        """
        if not self.masks:
            return None
            
        # Create a combined confidence map
        confidence_map = np.zeros((next(iter(self.masks.values())).shape))
        
        for cls_id, mask in self.masks.items():
            if cls_id in self.probs:
                prob = self.probs[cls_id]
                if isinstance(prob, torch.Tensor):
                    prob = prob.cpu().numpy()
                confidence_map = np.maximum(confidence_map, prob)
                
        return confidence_map
        
    def get_model_contribution(self, id_or_lbl):
        """
        Get the contribution of each model to the final segmentation for a class.
        
        Args:
            id_or_lbl: Class ID or label
            
        Returns:
            Dictionary of model contributions
        """
        cls_id = self.get_id(id_or_lbl)
        if cls_id is None:
            return None
            
        contributions = {}
        for model_name, model_data in self.model_results.items():
            if cls_id in model_data['masks']:
                mask = model_data['masks'][cls_id]
                if isinstance(mask, torch.Tensor):
                    mask_area = torch.sum(mask > 0).item()
                else:
                    mask_area = np.sum(mask > 0)
                contributions[model_name] = mask_area
                
        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            for model in contributions:
                contributions[model] /= total
                
        return contributions
    
    def process_image_numpy(self, image: np.ndarray, threshold=0.25):
        """Process a numpy image array."""
        # Print image shape for debugging
        print(f"Image shape in process_image_numpy: {image.shape}")
        
        # Ensure image is properly formatted
        if not isinstance(image, np.ndarray):
            print(f"Error: image is not a numpy array in process_image_numpy, it's {type(image)}")
            return None
            
        # Ensure uint8 format
        if image.dtype != np.uint8:
            print(f"Converting image from {image.dtype} to uint8")
            image = image.astype(np.uint8)
        
        return self.process_image(image, threshold=threshold)

    def Does(self, cv_image, threshold=0.25, 
                                  temporal_consistency=True, min_detections=2, 
                                  confidence_boost=0.25, max_frames=5):
        """
        Process an image with temporal consistency to improve detection reliability.
        
        Args:
            cv_image: OpenCV image to process
            threshold: Base confidence threshold for segmentation
            temporal_consistency: Whether to apply temporal consistency
            min_detections: Minimum number of detections required to consider an object valid
            confidence_boost: Amount to boost confidence for temporally consistent detections
            max_frames: Maximum number of frames to track for temporal consistency
            
        Returns:
            Segmentation results with temporal consistency applied
        """
        # Process the current frame normally
        self.frame_count += 1
        results = self.process_image(cv_image, threshold)
        
        if not temporal_consistency:
            return results
        
        # Track detections across frames
        current_frame_boxes = {}
        
        # Extract current frame's bounding boxes for each class
        for cls_id in self.masks:
            if cls_id not in current_frame_boxes:
                current_frame_boxes[cls_id] = []
                
            if cls_id in self.boxes:
                current_frame_boxes[cls_id].extend(self.boxes[cls_id])
        
        # Update temporal tracking with current detections
        for cls_id, boxes in current_frame_boxes.items():
            cls_name = self.id2label[cls_id]
            
            if cls_name not in self.temporal_tracking:
                self.temporal_tracking[cls_name] = []
                self.detection_history[cls_name] = {}
            
            # For each detected box in this frame
            for confidence, box in boxes:
                box_id = self._find_matching_box(cls_name, box)
                
                if box_id is None:
                    # New detection
                    box_id = f"{cls_name}_{len(self.detection_history[cls_name])}"
                    self.detection_history[cls_name][box_id] = {
                        'detections': 1,
                        'frames_seen': [self.frame_count],
                        'last_confidence': confidence,
                        'last_box': box
                    }
                else:
                    # Existing detection - update tracking
                    self.detection_history[cls_name][box_id]['detections'] += 1
                    self.detection_history[cls_name][box_id]['frames_seen'].append(self.frame_count)
                    self.detection_history[cls_name][box_id]['last_confidence'] = confidence
                    self.detection_history[cls_name][box_id]['last_box'] = box
                
                # Keep only recent frames
                if len(self.detection_history[cls_name][box_id]['frames_seen']) > max_frames:
                    self.detection_history[cls_name][box_id]['frames_seen'] = \
                        self.detection_history[cls_name][box_id]['frames_seen'][-max_frames:]
            
            # Update tracking list for this frame
            self.temporal_tracking[cls_name].append(self.frame_count)
            if len(self.temporal_tracking[cls_name]) > max_frames:
                self.temporal_tracking[cls_name] = self.temporal_tracking[cls_name][-max_frames:]
                
        # Apply temporal consistency boost to detection confidence
        for cls_name, boxes_dict in self.detection_history.items():
            for box_id, box_info in boxes_dict.items():
                # If we've seen this object multiple times
                if box_info['detections'] >= min_detections:
                    cls_id = self.label2id[cls_name]
                    
                    # Only boost if the object was detected in the current frame
                    if self.frame_count in box_info['frames_seen']:
                        confidence_multiplier = 1.0 + (confidence_boost * 
                                                min(1.0, box_info['detections'] / min_detections))
                        
                        # Find the index of this box in the current frame's detections
                        if cls_id in self.boxes:
                            for i, (conf, box) in enumerate(self.boxes[cls_id]):
                                if self._calculate_iou(box, box_info['last_box']) > 0.5:
                                    # Boost the confidence
                                    boosted_conf = min(0.99, conf * confidence_multiplier)
                                    self.boxes[cls_id][i] = (boosted_conf, box)
                                    
                                    # If we have masks, boost their confidence too
                                    if cls_id in self.masks and cls_id in self.probs:
                                        # Create a mask for just this box
                                        box_mask = np.zeros_like(self.masks[cls_id])
                                        x1, y1, x2, y2 = [int(coord) for coord in box]
                                        x1, y1 = max(0, x1), max(0, y1)
                                        try:
                                            box_mask[y1:y2, x1:x2] = 1
                                            # Only boost confidence within this box region
                                            if isinstance(self.probs[cls_id], torch.Tensor):
                                                box_mask_tensor = torch.tensor(box_mask, 
                                                                            device=self.probs[cls_id].device,
                                                                            dtype=self.probs[cls_id].dtype)
                                                # Apply boost only to the box region
                                                boost_mask = self.probs[cls_id] * box_mask_tensor * (confidence_multiplier - 1.0)
                                                self.probs[cls_id] = torch.clamp(self.probs[cls_id] + boost_mask, 0.0, 0.99)
                                            else:
                                                # NumPy array version
                                                boost_mask = self.probs[cls_id] * box_mask * (confidence_multiplier - 1.0)
                                                self.probs[cls_id] = np.clip(self.probs[cls_id] + boost_mask, 0.0, 0.99)
                                        except Exception as e:
                                            print(f"Error applying temporal boost: {e}")
        
        return results
    
    def _find_matching_box(self, cls_name, current_box, iou_threshold=0.5):
        """Find a matching box from previous detections."""
        if cls_name not in self.detection_history:
            return None
            
        for box_id, box_info in self.detection_history[cls_name].items():
            if self._calculate_iou(current_box, box_info['last_box']) > iou_threshold:
                return box_id
                
        return None
        
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
        # Convert boxes to [x1, y1, x2, y2] format if not already
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0  # No intersection
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate IoU
        return area_i / (area1 + area2 - area_i)


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
    parser.add_argument('--disable-temporal', action='store_true', help='disable temporal consistency')
    parser.add_argument('--min-detections', type=int, default=2, help='minimum detections for temporal consistency')
    parser.add_argument('--confidence-boost', type=float, default=0.25, help='confidence boost for temporal consistency')
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