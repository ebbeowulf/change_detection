#!/usr/bin/env python3
import numpy as np
import pdb
import tf
import copy

def get_world_pose(trans, quat):
    mm=tf.transformations.quaternion_matrix(quat)
    R=np.transpose(mm)
    pose=np.matmul(-R[:3,:3],trans)
    R[:3,3]=pose
    return pose, R

def plot_path(P, M, color_path=[0,1,0], color_direction=[1,0,0]):

    forward=np.matmul(M,[0,0,0.3,1])
    for idx in range(len(P)):
        plt.plot([P[idx,0],forward[idx,0]],[P[idx,1],forward[idx,1]],color=color_direction)
    plt.plot(P1[:,0],P1[:,1],color=color_path)

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

class image_set():
    def __init__(self, images_csv:str, fake_depth=None):
        self.read_image_csv(images_csv)
        self.set_fake_depth(fake_depth)
        # if set_fake_depth:
    
    def read_image_csv(self, images_txt):
        with open(images_txt,"r") as fin:
            A=fin.readlines()

        self.all_images={}
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
                # We will assume a format of {rootname}_{image_id}.png in the image name
                id_str=ln_s[-1].split('_')[1].split('.')[0]
                id=int(id_str)
                quat=[float(ln_s[2]),float(ln_s[3]), float(ln_s[4]), float(ln_s[1])]
                trans=[float(ln_s[5]),float(ln_s[6]),float(ln_s[7])]
                world_pose, rotM = get_world_pose(trans, quat)
                image={'rot': quat, 'trans': trans, 'id': id, 'global_pose': world_pose, 'global_poseM': rotM, 'name': ln_s[-1]}
                self.all_images[ln_s[-1]]=image

    def set_fake_depth(self, fixed_depth:float):
        # Z is forward in the camera frame
        V=np.array([0,0,fixed_depth,1.0])        
        for key in self.all_images.keys():
            self.all_images[key]['center_pose']=np.matmul(self.all_images[key]['global_poseM'],V)

    def get_pose_by_name(self, im_name):
        if key in self.all_images:
            return self.all_images[key]['center_pose']
        return None

    def get_pose_by_id(self, id):
        for key in self.all_images.keys():
            if self.all_images[key]['id']==id:
                return self.all_images[key]['center_pose']
        return None

    def get_related_poses(self, tgt_pose:np.array, max_dist:float=2):
        im_list=[]
        for key in self.all_images.keys():
            im = self.all_images[key]
            # Filter Images by Distance from Target
            deltaD=dist(tgt_pose-im['center_pose'])
            if deltaD>max_dist:
                continue

            im_list.append(key)
        
        return im_list
    
    def get_all_poses(self):
        arr = []
        for key in self.all_images.keys():
            arr.append(self.all_images[key]['global_pose'])
        return np.array(arr)

