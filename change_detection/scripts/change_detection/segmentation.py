import numpy as np
import pdb
from sklearn.cluster import DBSCAN
import cv2

class image_segmentation():
    def __init__(self):
        print("Base model image segmentation")
        self.label2id = {} # format label: id
        self.id2label = {} # format id: label
        self.clear_data()

    def clear_data(self):
        self.masks = {}     # store each mask as a separate item in the dictionary, organized by
                            # id, so that way some masks can be empty if not applicable
        self.probs = {}     # store each probability as an array per label - again as a dictionary
        self.max_probs = {} # store each probability as a maximum per label - again as a dictionary
        self.boxes = {}     # for each class, store a list of boxes and their associated probabilities

    # Process an image file - implemented in the sub-classes
    #   fName       - file containing the image
    #   threshold   - detection threshold to apply if applicable
    #   save_fileName - save the resulting intermediate file? (default = no)
    def process_file(self, fName:str, threshold:float, save_fileName:str=None):
        print("Base: process_file")
    
    def compute_dynamic_threshold(self, image, base_threshold=0.5, min_threshold=0.3, max_threshold=0.7):
        """
        Compute a dynamic threshold based on image brightness.
        
        Args:
            image (np.ndarray): Input image in BGR format.
            base_threshold (float): Default threshold to adjust (e.g., 0.5).
            min_threshold (float): Minimum allowable threshold (e.g., 0.3).
            max_threshold (float): Maximum allowable threshold (e.g., 0.7).
        
        Returns:
            float: Dynamic threshold value.
        """
        # Convert to grayscale for brightness calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute mean intensity (brightness)
        mean_intensity = np.mean(gray)
        
        # Normalize brightness to [0, 1]
        normalized_intensity = mean_intensity / 255.0
        
        # Adjust threshold based on brightness
        scale_factor = 0.2  # Controls sensitivity to brightness changes
        dynamic_threshold = base_threshold + (normalized_intensity - 0.5) * scale_factor
        
        # Clamp to min and max thresholds
        dynamic_threshold = max(min_threshold, min(max_threshold, dynamic_threshold))
        
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (gray.shape[0] * gray.shape[1])
        if edge_density > 0.1:
            dynamic_threshold = base_threshold + 0.1

        return dynamic_threshold

    # Load an existing results file - returns True if successful
    def load_file(self, fileName):
        print("Base: load file.")
        return False

    def get_all_classes(self):
        # Get all the classes
        return self.label2id

    def get_id(self, id_or_lbl):
        if self.label2id is None:
            return None
        if type(id_or_lbl)==str:
            if id_or_lbl in self.label2id:
                return self.label2id[id_or_lbl]
            else:
                print(id_or_lbl + "not in valid list of classes")
                return None
        elif type(id_or_lbl)==int and id_or_lbl in self.id2label:
            return id_or_lbl

    def get_mask(self, id_or_lbl):
        id = self.get_id(id_or_lbl)
        if id is not None and id in self.masks:
            return self.masks[id]
        return None
    
    def get_max_prob(self, id_or_lbl):
        id = self.get_id(id_or_lbl)
        if id is not None and id in self.max_probs:
            return self.max_probs[id]
        return None

    def get_prob_array(self, id_or_lbl):
        id = self.get_id(id_or_lbl)
        if id is not None and id in self.probs:
            return self.probs[id]
        return None

    # Get saved boxes associated with the target
    def get_boxes(self, id_or_lbl):
        id = self.get_id(id_or_lbl)
        if id is not None and id in self.boxes:
            return self.boxes[id]

    # a method for building boxes from raw probability data by clustering the specified points
    #   this will erase previously stored boxes 
    def build_dbscan_boxes(self, cls_target, threshold, eps=10, min_samples=100, MAX_CLUSTERING_SAMPLES=50000):
        key = self.get_id(cls_target)
        if key is None:
            return
        self.boxes[key]=[]
        if self.max_probs[key]<threshold:            
            return
        
        rows,cols=np.nonzero(self.masks[0])
        xy_grid_pts=np.vstack((cols,rows)).transpose()
        scores=self.probs[key][rows,cols]
        if xy_grid_pts is None or xy_grid_pts.shape[0]<min_samples:
            return

        # Need to contrain the maximum number of points - else dbscan will be extremely slow
        if xy_grid_pts.shape[0]>MAX_CLUSTERING_SAMPLES:        
            rr=np.random.choice(np.arange(xy_grid_pts.shape[0]),size=MAX_CLUSTERING_SAMPLES)
            xy_grid_pts=xy_grid_pts[rr]
            scores=scores[rr]

        CL2=DBSCAN(eps=eps, min_samples=min_samples).fit(xy_grid_pts,sample_weight=scores)
        for idx in range(10):
            whichP=np.where(CL2.labels_== idx)            
            if len(whichP[0])<1:
                break
            box=np.hstack((xy_grid_pts[whichP].min(0),xy_grid_pts[whichP].max(0)))
            self.boxes[key].append((scores[whichP].max(),box))
