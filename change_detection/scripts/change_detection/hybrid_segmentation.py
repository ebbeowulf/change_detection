# In change_detection/scripts/change_detection/hybrid_segmentation.py
from change_detection.image_segmentation import image_segmentation
from change_detection.omdet_segmentation import omdet_segmentation
from change_detection.clip_segmentation import clip_segmentation
import torch
import numpy as np
from PIL import Image
import pickle
import pdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hybrid_segmentation(image_segmentation):
    def __init__(self, prompts):
        
        # Initialize both detectors
        self.omdet_detector = omdet_segmentation(prompts)
        self.clipseg_detector = clip_segmentation(prompts)  # Changed to clip_segmentation
        
        # Common parameters
        self.prompts = prompts
        self.clear_data()
        
    def clear_data(self):
        self.masks = {}
        self.boxes = {}
        self.scores = {}
        self.labels = {}
        self.sources = {}  # Track which detector found each object
    
    def process_image(self, image, threshold=0.25):
        """Process image with both models and combine results"""
        self.clear_data()
        
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
            
        # Process with both detectors
        omdet_results = self.omdet_detector.process_image(image_pil, threshold)
        clipseg_results = self.clipseg_detector.process_image(image_pil, threshold)
        
        # Combine results from both detectors
        combined_results = self._combine_results(omdet_results, clipseg_results)
        
        if combined_results:
            self.set_data(combined_results)
            return combined_results
        else:
            print("No objects detected by either detector.")
            return None
    
    def process_image_numpy(self, image: np.ndarray, threshold=0.25):
        """Process numpy image with both detectors"""
        image_pil = Image.fromarray(image)
        return self.process_image(image_pil, threshold)
    
    def _combine_results(self, omdet_results, clipseg_results):
        """Combine results from both detectors, handling different output formats"""
        # Handle the case where one or both detectors return nothing
        if not omdet_results and not clipseg_results:
            return None
            
        # CLIPSeg doesn't return the same structure as OmDet-Turbo with SAM
        # It returns masks directly, so we need to adapt the combination logic
        
        # Start with OmDet results if available
        if omdet_results:
            combined = {}
            # Copy over the OmDet masks, boxes, etc.
            for prompt_idx, prompt in enumerate(self.prompts):
                if prompt in self.omdet_detector.masks:
                    if prompt not in combined:
                        combined[prompt] = {
                            'masks': [], 'boxes': [], 'scores': [], 'sources': []
                        }
                    
                    # Add all OmDet detections for this prompt
                    combined[prompt]['masks'].extend(self.omdet_detector.masks[prompt])
                    combined[prompt]['boxes'].extend(self.omdet_detector.boxes[prompt])
                    combined[prompt]['scores'].extend(self.omdet_detector.scores[prompt])
                    combined[prompt]['sources'].extend(['omdet'] * len(self.omdet_detector.masks[prompt]))
        else:
            combined = {}
        
        # Add CLIPSeg results if available
        if clipseg_results:
            for prompt_idx, prompt in enumerate(self.prompts):
                if prompt in self.clipseg_detector.masks:
                    if prompt not in combined:
                        combined[prompt] = {
                            'masks': [], 'boxes': [], 'scores': [], 'sources': []
                        }
                    
                    # Add all CLIPSeg detections for this prompt
                    combined[prompt]['masks'].extend(self.clipseg_detector.masks[prompt])
                    combined[prompt]['boxes'].extend(self.clipseg_detector.boxes[prompt])
                    combined[prompt]['scores'].extend(self.clipseg_detector.scores[prompt])
                    combined[prompt]['sources'].extend(['clipseg'] * len(self.clipseg_detector.masks[prompt]))
        
        return combined
    
    def set_data(self, combined_results):
        """Set internal data from combined results dictionary"""
        self.masks = {}
        self.boxes = {}
        self.scores = {}
        self.sources = {}
        
        pdb.set_trace()
        for prompt, data in combined_results.items():
            self.masks[prompt] = data['masks']
            self.boxes[prompt] = data['boxes']
            self.scores[prompt] = data['scores']
            self.sources[prompt] = data['sources']