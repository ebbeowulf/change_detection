import copy
import numpy as np

class rgbd_file_list():
    def __init__(self, color_image_dir:str, depth_image_dir:str, intermediate_save_dir:str):
        self.all_files=dict()
        self.color_image_dir=color_image_dir + "/"
        self.depth_image_dir=depth_image_dir + "/"
        self.intermediate_save_dir=intermediate_save_dir + "/"
    
    def add_file(self, id:int, color_fName:str, depth_fName:str):
        if id in self.all_files:
            print("ID %d is already in the file list - replacing existing entry")

        self.all_files[id]={'color': color_fName, 'depth': depth_fName}

    def add_pose(self, id:int, pose:np.array):
        self.all_files[id]['pose']=pose

    def keys(self):
        return self.all_files.keys()
    
    def is_key(self, id:int):
        return id in self.all_files
    
    def get_color_fileName(self, id:int):
        return self.color_image_dir+self.all_files[id]['color']

    def get_segmentation_fileName(self, id:int, is_yolo:bool, tgt_class:str):
        if is_yolo:
            return self.get_yolo_fileName(id)
        else:
            return self.get_clip_fileName(id, tgt_class)
        
    def get_yolo_fileName(self, id:int):
        return self.intermediate_save_dir+self.all_files[id]['color']+".yolo.pkl"

    def get_clip_fileName(self, id:int, tgt_class:str):
        cls_str=copy.copy(tgt_class)
        cls_str=cls_str.replace(" ","_")
        return self.intermediate_save_dir+self.all_files[id]['color']+".%s.clip.pkl"%(cls_str)

    def get_depth_fileName(self, id:int):
        return self.depth_image_dir+self.all_files[id]['depth']
    
    def get_pose(self, id:int):
        return self.all_files[id]['pose']
    
    def get_class_pcloud_fileName(self, id:int, cls:str):
        return self.intermediate_save_dir+self.all_files[id]['color']+".%s.pkl"%(cls)

    def get_combined_pcloud_fileName(self, cls:str=None):
        if cls==None:
            return self.intermediate_save_dir+"combined.ply"
        else:
            return self.intermediate_save_dir+"%s.ply"%(cls)

    def get_combined_raw_fileName(self, cls:str):
        return self.intermediate_save_dir+cls+".raw.pkl"

    def get_annotation_file(self):
        return self.intermediate_save_dir+"annotations.json"
    
    def get_labeled_pcloud_fileName(self, cls:str):
        return self.intermediate_save_dir+"%s.labeled.ply"%(cls)

    def get_json_summary_fileName(self, cls:str):
        return self.intermediate_save_dir+"%s.summary.json"%(cls)
    