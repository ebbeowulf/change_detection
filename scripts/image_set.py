#!/usr/bin/env python3
import numpy as np
import pdb
import tf
import copy

def read_image_csv(images_txt):
    with open(images_txt,"r") as fin:
        A=fin.readlines()

    all_images=[]
    for ln_ in A:
        if len(ln_)<2:
            continue
        if ln_[-1]=='\n':
            ln_=ln_[:-1]
        if ln_[0]=='#':
            continue
        if ',' in ln_:
            ln_s=ln_.split(', ')
        else:
            ln_s=ln_.split(' ')

        if len(ln_s)==10:
            quat=[float(ln_s[2]),float(ln_s[3]), float(ln_s[4]), float(ln_s[1])]
            trans=[float(ln_s[5]),float(ln_s[6]),float(ln_s[7])]
            image={'rot': quat, 'trans': trans, 'name': ln_s[-1], 'id': int(ln_s[0])}
            all_images.append(image)
    return all_images


class create_image_vector():
    def __init__(self, clip_csv:str):
        with open(clip_csv,'r') as fin:
            A=fin.readlines()
        
        # Remove eol characters
        for i in range(len(A)):
            if A[i][-1]=='\n':
                A[i]=A[i][:-1]
        
        self.labels_=A[0].split(', ')[1:]

        self.results={}
        for ln in A[1:]:
            lnS=ln.split(', ')
            arr = np.zeros((len(lnS)-1,1))
            for idx, val in enumerate(lnS[1:]):
                arr[idx]=float(val)
            self.results[lnS[0]]=arr
    
    def get_labels(self):
        return copy.copy(self.labels_)
    def get_array(self, image_list:list):
        arr=None
        for im in image_list:
            if im in self.results:
                if arr is None:
                    arr = self.results[im]
                else:
                    arr = np.hstack((arr,self.results[im]))
        return arr

def dist(A:np.array):
    return np.sqrt(np.power(A,2).sum())

class get_neighboring_images():
    def __init__(self, images_csv:str):
        self.all_images=read_image_csv(images_csv)

    def get_pose_by_name(self, im_name, dist=1.5):
        trans=None
        rot=None
        for im in self.all_images:
            if im['name']==im_name:
                trans=im['trans']
                rot=im['rot']
                break
        base_relativeM=np.matmul(tf.transformations.translation_matrix(trans),tf.transformations.quaternion_matrix(rot))
        linear=[dist,0,0,1.0]
        tgt_pose=np.matmul(base_relativeM,linear)
        return tgt_pose

    def get_related_poses(self, tgt_pose:np.array, max_dist:float=2, angle_dist:float=0.5):
        im_list=[]
        for im in self.all_images:
            # Filter Images by Distance from Target
            deltaD=np.sqrt(np.power(im['trans']-tgt_pose[:3],2).sum())
            if deltaD>max_dist:
                continue

            # Create a vector representing the direction of the camera
            M1=tf.transformations.quaternion_matrix(im['rot'])
            V1=np.matmul(M1,[1,0,0,1])[:3]
            V2=tgt_pose[:3]-np.array(im['trans'])
            angle=np.arccos(np.dot(V1,V2)/(dist(V1)*dist(V2)))

            if angle>angle_dist:
                continue

            im_list.append(im)
        
        return im_list
    
    def get_all_poses(self):
        arr=[self.all_images[0]['trans']]
        for im in self.all_images[1:]:
            arr.append(im['trans'])
        return np.array(arr)

