#!/usr/bin/env python3
import numpy as np
import pdb
import copy
from scipy.spatial.transform import Rotation as R

def read_depthP_csv(fName):
    with open(fName,"r") as fin:
        A=fin.readlines()

    all_images={}
    for ln_ in A:        
        if len(ln_)<2 or 'PX' in ln_:
            continue
        if ln_[-1]=='\n':
            ln_=ln_[:-1]
        ln_s=ln_.split(', ')
        # Should be of format:
        #   image, PX, PY, PZ, QX, QY, QZ, QW, CenterPX, CenterPY, CenterPZ
        name=ln_s[0]
        id_str=name.split('_')[1].split('.')[0]
        id=int(id_str)
        world_pose=np.array([float(ln_s[1]),float(ln_s[2]),float(ln_s[3])])
        quat=[float(ln_s[4]),float(ln_s[5]),float(ln_s[6]),float(ln_s[7])]
        rotM=np.identity(4)
        rotM[:3,:3]=R.from_quat(quat).as_matrix()
        rotM[:3,3]=world_pose
        center_pose=np.array([float(ln_s[8]),float(ln_s[9]),float(ln_s[10])])
        all_images[name]={'id': id, 'global_pose': world_pose, 'global_poseM': rotM, 'name': name, 'center_pose': center_pose}
    return all_images
    
def dist(A:np.array):
    return np.sqrt(np.power(A,2).sum())

class create_image_vector():
    def __init__(self, clip_csv:str):
        with open(clip_csv,'r') as fin:
            A=fin.readlines()
        
        # Remove eol characters
        for i in range(len(A)):
            if A[i][-1]=='\n':
                A[i]=A[i][:-1]
        
        self.labels_=np.array(A[0].split(', ')[1:])
        self.set_label_mask([],True)

        self.results={}
        for ln in A[1:]:
            lnS=ln.split(', ')
            arr = np.zeros((len(lnS)-1,1))
            for idx, val in enumerate(lnS[1:]):
                arr[idx]=float(val)
            self.results[lnS[0]]=arr
    
    def build_mean_vector(self):
        arr=np.zeros((self.labels_.shape[0],len(self.results.keys())),dtype=float)
        for idx, key in enumerate(self.results.keys()):
            arr[:,idx]=self.results[key]
        return arr.mean(1)
        
    def set_label_mask(self, object_list:list, set_base_mask=False):
        self.object_label_mask=[]
        self.base_label_mask=[]
        for idx, lbl in enumerate(self.labels_):
            if set_base_mask:
                if lbl in object_list:
                    self.base_label_mask.append(idx)
                else:
                    self.object_label_mask.append(idx)
            else:
                if lbl in object_list:
                    self.object_label_mask.append(idx)
                else:
                    self.base_label_mask.append(idx)

    def get_vector(self, image_name:str, use_base_mask=False):
        if image_name in self.results:
            if use_base_mask:
                return self.results[image_name][self.base_label_mask]
            else:
                return self.results[image_name][self.object_label_mask]
        return None

    def get_array(self, image_list:list, use_base_mask=False):
        arr=None
        for im in image_list:
            if im in self.results:
                V=self.get_vector(im,use_base_mask)
                if V is None:
                    continue
                if arr is None:
                    arr = V
                else:
                    arr = np.hstack((arr,V))
        return arr

    def create_gaussian_model(self, image_list, use_base_mask=False):
        try:
            arr=self.get_array(image_list, use_base_mask)
            cov=np.cov(arr)
            model={'mean': arr.mean(1).reshape((arr.shape[0],1)),
                    'cov': cov, 
                    'stdev': arr.std(1),
                    'inv_cov': np.linalg.inv(cov) }
            return model
        except Exception as e:
            return None

    def get_labels(self, use_base_mask=False):
        if use_base_mask:
            return self.labels_[self.base_label_mask]
        else:
            return self.labels_[self.object_label_mask]


class image_set():
    def __init__(self, images_csv:str):
        self.all_images = read_depthP_csv(images_csv)
        # self.set_fake_depth(fake_depth)
        # if set_fake_depth:

    # def set_fake_depth(self, fixed_depth:float):
    #     # Z is forward in the camera frame
    #     V=np.array([0,0,fixed_depth,1.0])        
    #     for key in self.all_images.keys():
    #         self.all_images[key]['center_pose']=np.matmul(self.all_images[key]['global_poseM'],V)

    def get_pose_by_name(self, key):
        if key in self.all_images:
            return self.all_images[key]['center_pose']
        return None

    def get_pose_by_id(self, id):
        for key in self.all_images.keys():
            if self.all_images[key]['id']==id:
                return self.all_images[key]['center_pose']
        return None

    def get_related_poses(self, tgt_pose:np.array, max_dist:float=1, max_angle:float=0.5):
        im_list=[]
        for key in self.all_images.keys():
            im = self.all_images[key]
            # Filter Images by Distance from Target
            deltaD=dist(tgt_pose-im['center_pose'])
            if deltaD>max_dist:
                continue

            forward=im['center_pose']-im['global_pose']
            actual=tgt_pose-im['global_pose']
            angle=np.arccos(np.dot(forward,actual)/(dist(forward)*dist(actual)))
            if np.fabs(angle)<max_angle:
                im_list.append(key)

        return im_list
    
    def get_all_poses(self):
        arr = []
        for key in self.all_images.keys():
            arr.append(self.all_images[key]['global_pose'])
        return np.array(arr)

