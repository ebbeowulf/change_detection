from change_detection.image_segmentation import image_segmentation
from change_detection.omdet_segmentation import omdet_segmentation
from change_detection.clip_segmentation import clip_segmentation
import torch
import numpy as np
from PIL import Image
import copy
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hybrid_segmentation(image_segmentation):
    def __init__(self, prompts):
        print("Initializing hybrid OmDet-Turbo + CLIPSeg detector")
        
        # Initialize both detectors
        self.omdet_detector = omdet_segmentation(prompts)
        self.clipseg_detector = clip_segmentation(prompts)
        
        # Common parameters
        self.prompts = prompts
        self.clear_data()
        
    def clear_data(self):
        self.masks = {}
        self.boxes = {}
        self.scores = {}
        self.sources = {}
    
    def process_image(self, image, threshold=0.25):
        """Process image with both models"""
        self.clear_data()
        
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        # Process with both detectors
        self.omdet_detector.clear_data()
        omdet_results = self.omdet_detector.process_image(image_pil, threshold)
        
        self.clipseg_detector.clear_data()
        clipseg_results = self.clipseg_detector.process_image(image_pil, threshold)
        
        # Store detection data for all prompts
        for prompt in self.prompts:
            if prompt not in self.masks:
                self.masks[prompt] = []
                self.boxes[prompt] = []
                self.scores[prompt] = []
                self.sources[prompt] = []
            
            # Add OmDet detections
            if prompt in self.omdet_detector.masks:
                self.masks[prompt].extend(self.omdet_detector.masks[prompt])
                self.boxes[prompt].extend(self.omdet_detector.boxes[prompt])
                self.scores[prompt].extend(self.omdet_detector.scores[prompt])
                self.sources[prompt].extend(['omdet'] * len(self.omdet_detector.masks[prompt]))
            
            # Add CLIPSeg detections
            if prompt in self.clipseg_detector.masks:
                self.masks[prompt].extend(self.clipseg_detector.masks[prompt])
                self.boxes[prompt].extend(self.clipseg_detector.boxes[prompt])
                self.scores[prompt].extend(self.clipseg_detector.scores[prompt])
                self.sources[prompt].extend(['clipseg'] * len(self.clipseg_detector.masks[prompt]))
        
        # Check if we have any detections
        has_detections = False
        for prompt in self.prompts:
            if prompt in self.masks and len(self.masks[prompt]) > 0:
                has_detections = True
                break
                
        if has_detections:
            return True  # Successfully processed
        else:
            print("No objects detected by either detector.")
            return None
    
    def process_image_numpy(self, image: np.ndarray, threshold=0.25):
        """Process numpy image with both detectors"""
        image_pil = Image.fromarray(image)
        return self.process_image(image_pil, threshold)
        
    def get_detection_sources_for_prompt(self, prompt):
        """Helper to check which sources detected a given prompt"""
        if prompt not in self.sources or not self.sources[prompt]:
            return []
        return list(set(self.sources[prompt]))
        
    def get_mask(self, prompt, source=None):
        """Get masks for a specific prompt and optionally filter by source"""
        if prompt not in self.masks:
            return None
            
        if source is None:
            # Return all masks for this prompt
            return self.masks[prompt]
        else:
            # Filter masks by source
            source_masks = []
            for i, src in enumerate(self.sources[prompt]):
                if src == source:
                    source_masks.append(self.masks[prompt][i])
            
            if not source_masks:
                return None
            
            # Return concatenated masks
            return source_masks
            
    def get_prob_array(self, prompt, source=None):
        """Get probability arrays for a specific prompt and optionally filter by source"""
        if prompt not in self.scores:
            return None
            
        if source is None:
            # Return all scores for this prompt
            return self.scores[prompt]
        else:
            # Filter scores by source
            source_scores = []
            for i, src in enumerate(self.sources[prompt]):
                if src == source:
                    source_scores.append(self.scores[prompt][i])
            
            if not source_scores:
                return None
            
            # Return concatenated scores
            return source_scores

    def load_file(self, fileName, threshold=0.5):
        """Load from a saved file (implementation depends on your data structure)"""
        try:
            # Try to load the file
            with open(fileName, 'rb') as handle:
                save_data = pickle.load(handle)
                self.clear_data()
                if save_data['prompts'] == self.prompts:
                    # Handle loading data - this depends on file format
                    if 'outputs' in save_data:
                        self.set_data(save_data['outputs'])
                    return True
                else:
                    print("Prompts in saved file do not match... skipping")
        except Exception as e:
            print(f"Error loading file: {e}")
        return False
