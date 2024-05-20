import numpy as np
import pdb

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

    def process_file(self, fName):
        print("Base: process string")

    def process_image(self, cv_image):
        print("Base: process_image.")
    
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
    