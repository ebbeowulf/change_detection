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
        self.set_base_label_mask([])

        self.results={}
        for ln in A[1:]:
            lnS=ln.split(', ')
            arr = np.zeros((len(lnS)-1,1))
            for idx, val in enumerate(lnS[1:]):
                arr[idx]=float(val)
            self.results[lnS[0]]=arr
    
    def set_base_label_mask(self, base_list:list):
        self.object_label_mask=[]
        self.base_label_mask=[]
        for idx, lbl in enumerate(self.labels_):
            if lbl in base_list:
                self.base_label_mask.append(idx)
            else:
                self.object_label_mask.append(idx)

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
        arr=self.get_array(image_list, use_base_mask)
        cov=np.cov(arr)
        return {'mean': arr.mean(1).reshape((arr.shape[0],1)),
                'cov': cov, 
                'stdev': arr.std(1),
                'inv_cov': np.linalg.inv(cov) }

    def get_labels(self):
        return copy.copy(self.labels_)


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

