import numpy as np
import pdb
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
import glob
import pickle
from change_pcloud_utils.llm_utils import before_and_after_cluster_filter, multi_view_cluster_filter
from change_pcloud_utils.map_utils import identify_related_images_global_pose
from PIL import Image
import torch

def gaussian_dist_function(val1, std1):
    return np.exp(-0.5*np.power(val1/std1,2))

def get_values_from_dict(d_in, tgt_key, default_val):
    p_array=[]
    for key in d_in:
        if tgt_key in d_in[key]:
            p_array.append(d_in[key][tgt_key])
        else:
            p_array.append(default_val)
    return p_array

# The primary purpose of the general class here
# is to provide a common interface for a number of different 
# filtering approaches. The filter also allows us to save
# the results for each generated set of clusters - which
# is important for slow methods like those using LLMs
class cluster_filter():
    def __init__(self):
        self.clear_results()
        
    def clear_results(self):
        self.scores=dict() # to be stored as 'cluster id': float

    def get_save_file_name(self, tgt_dir, prompt):
        return f"{tgt_dir}/{self.__class__.__name__}-{prompt}.json"

    def save2file(self, tgt_dir, prompt):
        fileName=self.get_save_file_name(tgt_dir, prompt)
        filter_state=self.serialize()
        with open(fileName,'w') as fout:
            json.dump(filter_state,fout)

    def serialize(self):
        return {'type': self.__class__.__name__, 'scores': self.scores}

    def loader(self, json_dict):
        assert(json_dict['type']==self.__class__.__name__)
        self.scores=json_dict['scores']

    def loadFromFile(self, tgt_dir, prompt):
        fileName=self.get_save_file_name(tgt_dir, prompt)
        if os.path.exists(fileName):
            try:
                with open(fileName,'r') as fin:
                    A=json.load(fin)
                self.loader(A)
            except Exception as e:
                print(f"{fileName} not found")
                return

    def save_score(self, cluster_id:str, score):
        self.scores[cluster_id] = score

    def calculate_score(self, all_images, cluster=None):
        raise Exception("Called the baseline calculate_score function - ERROR")

    def get_score(self, cluster_id:str, all_images=None, cluster=None):
        assert(type(cluster_id)==str)
        if cluster_id not in self.scores:
            if all_images is None:
                raise Exception("Cannot score cluster without a valid all_images struct")
            self.scores[cluster_id]=self.calculate_score(all_images, cluster)
        return self.scores[cluster_id]

#################
# prob_mean_filter
#   Scores the max probability of the object per image that should be looking at the cluster
#   It uses a default detection threshold for images with no detections
#################
class prob_mean_filter(cluster_filter):
    def __init__(self, segmentation_detection_threshold=0.1):
        self.detection_threshold=segmentation_detection_threshold
        super().__init__()
    
    def calculate_score(self, all_images, cluster=None):
        pn = np.array(get_values_from_dict(all_images, 'max_prob',self.detection_threshold))
        return pn.mean() 

#################
# pct_valid_filter
#   Tracks the percentage of images with points inside the bounding box
#   to those with no detections. 
#################
class pct_valid_filter(cluster_filter):
    def __init__(self):
        super().__init__()
    
    def calculate_score(self,all_images, cluster=None):
        valid_images=np.array([ 'new' in all_images[key] for key in all_images ])
        return (valid_images.sum()/len(all_images.keys()))  

#################
# cluster_stat_filter
#   Returns one of the cluster stats
#   stat_name: max, mean, prob_sum, pcount, stdev
#################
class cluster_stat_filter(cluster_filter):
    def __init__(self, stat_name:str):
        valid_names=['max','mean','prob_sum','pcount','stdev']
        if stat_name not in valid_names:
            raise Exception("Invalid stat name to retrieve")
        self.stat=stat_name
        super().__init__()
    
    def calculate_score(self,all_images, cluster):
        if cluster is None:
            raise Exception("Must provide cluster for score calculations")
        val=cluster.prob_stats[self.stat]
        if type(val)==torch.Tensor:
            val=val.cpu().item()
        return val

#################
# combo filter
#################
class combo_filter(cluster_filter):
    def __init__(self, filter1:cluster_filter, filter2: cluster_filter, operator:str="*", score_range:list=[0.05,0.99]):
        self.filter1=filter1
        self.filter2=filter2
        self.score_range=score_range
        assert(operator in ["*","+","lo"])
        self.operator=operator
        super().__init__()
    
    def combine_scores(self, score1, score2):
        score1_redux=min(self.score_range[1],max(self.score_range[0],score1))
        score2_redux=min(self.score_range[1],max(self.score_range[0],score2))
        if self.operator=="*":
            return score1_redux*score2_redux
        elif self.operator=="+":
            return score1_redux+score2_redux
        elif self.operator=="lo": #log-odds
            LO=np.log(score1_redux/(1-score1_redux))+np.log(score2_redux/(1-score2_redux))
            return np.exp(LO)/(1+np.exp(LO))
        return None        
        
    def calculate_score(self, all_images, cluster=None):
        return self.combine_scores(self.filter1.calculate_score(all_images, cluster),
                                   self.filter2.calculate_score(all_images, cluster))

    def loadFromFile(self, tgt_dir, prompt):
        self.filter1.loadFromFile(tgt_dir, prompt)
        self.filter2.loadFromFile(tgt_dir, prompt)
        for key in self.filter1.scores.keys():
            if key in self.filter2.scores:
                self.scores[key]=self.combine_scores(self.filter1.get_score(key),self.filter2.get_score(key))
            else:
                print(f"Key {key} missing from filter2")

        for key in self.filter2.scores.keys():
            if key not in self.scores:
                print(f"Key {key} missing from filter1")

    def save2file(self, tgt_dir, prompt):
        self.filter1.save2file(tgt_dir, prompt)
        self.filter2.save2file(tgt_dir, prompt)

#################
# LLM - Before And After Filter
#   Tracks the percentage of images with points inside the bounding box
#   to those with no detections. 
#################
class before_and_after_filter(cluster_filter):
    def __init__(self, fList_base, fList_new, params, use_render=True):
        self.use_render=use_render
        self.fList_new=fList_new
        self.fList_base=fList_base
        self.params=params
        if self.use_render:
            self.ba_cluster_filt=before_and_after_cluster_filter()
            self.before_key='render'
        else:
            desc=['The attached image includes two images of the same location captured at different times.',
                    'A blue box is drawn around the object of interest in one image.',
                    '(1) Is the object in the blue box not found in both images?',
                    '(2) Should a housekeeper pick up the object in the blue box and put it away?']
            self.ba_cluster_filt=before_and_after_cluster_filter(desc)
            self.before_key='before'
        super().__init__()
    
    def set_closest_images(self, all_images, cluster_centroid):
        # find images in the baseline dataset that can see the target cluster
        rel_imgs=identify_related_images_global_pose(self.params, self.fList_base, cluster_centroid, None, 0.5)

        # now find the vector to the object and the associated distance
        dist_relM=np.zeros((len(rel_imgs)))
        zVec_relM=np.zeros((3,len(rel_imgs)))
        for key_idx, rel_key in enumerate(rel_imgs):
            relM=np.matmul(params.rot_matrix,fList_base.get_pose(int(rel_key)))
            dist_relM[key_idx]=np.sqrt(((cluster_centroid[:3]-relM[:3,3])**2).sum())
            zVec_relM[:,key_idx]=np.matmul(relM[:3,:3],[0,0,1])
        zVec_relM=np.transpose(zVec_relM)

        # step through images that point at the object and calculate     
        for key in all_images.keys():
            if 'new' in all_images[key]:
                M=np.matmul(params.rot_matrix,self.fList_new.get_pose(int(key)))
                distM=np.sqrt(((cluster_centroid-M[:3,3])**2).sum())
                zVecM=np.matmul(M[:3,:3],[0,0,1])
                angle=zVec_relM@zVecM #dot-product, returns a [N,] vector
                best_match=rel_imgs[np.argmax(gaussian_dist_function(dist_relM-distM,1.0)*gaussian_dist_function(angle,0.25))]
                color_fName=fList_base.get_color_fileName(int(best_match))
                try:
                    img=Image.open(color_fName)
                    all_images[key]['before']=np.array(img)
                except Exception as e:
                    print(f"Cannot load image {color_fName}- skipping")
        return all_images
    
    def calculate_score(self,all_images, cluster=None):
        if not self.use_render: # need to add "before" images to all_images
            all_images=self.set_closest_images(all_images,cluster.centroid)
        # Need to flip dimensions of the images - they are inverted
        for key in all_images.keys():
            if 'new' in all_images[key]:
                all_images[key]['new']=all_images[key]['new'][:,:,[2,1,0]]
            if 'render' in all_images[key]:
                all_images[key]['render']=all_images[key]['render'][:,:,[2,1,0]]
        self.last_result=dict()
        pickup_pct, new_pct=self.ba_cluster_filt.evaluate_multiple_image_pairs(all_images, before_key=self.before_key)
        self.last_result={'is_pickup':pickup_pct, 'is_new':new_pct}
        return new_pct

    def get_save_file_name(self, tgt_dir, prompt):
        if self.use_render:
            return f"{tgt_dir}/render_and_after_filter-{prompt}.json"
        return super().get_save_file_name(tgt_dir, prompt)
    
    def clear_results(self):
        self.all_results=dict()
        return super().clear_results()
    
    def get_score(self, cluster_id, all_images=None, cluster=None):
        if cluster_id not in self.scores:
            if all_images is None:
                raise Exception("Cannot score cluster without a valid all_images struct")
            self.scores[cluster_id]=self.calculate_score(all_images, cluster)
            self.all_results[cluster_id]=self.last_result
        return self.scores[cluster_id]
    
    def serialize(self):
        filter_state=super().serialize()
        filter_state['all_results']=self.all_results
        return filter_state
    
    def loader(self, json_dict):
        super().loader(json_dict)
        self.all_results=json_dict['all_results']

#################
# is_pickup_filter - query llm about picking up objects specifically
#################
class is_pickup_filter(cluster_filter):
    def __init__(self, image_scale=1.0):
        self.llm_obj=multi_view_cluster_filter(image_scale)
        super().__init__()
    
    def calculate_score(self, all_images, cluster=None):
        return self.llm_obj.evaluate_cluster(all_images)

    def clear_results(self):
        self.all_results=dict()
        return super().clear_results()

    def get_save_file_name(self, tgt_dir, prompt):
        if self.llm_obj.scale!=1.0:
            return f"{tgt_dir}/is_pickup_filter{self.llm_obj.scale}-{prompt}.json"
        return super().get_save_file_name(tgt_dir, prompt)

    def get_score(self, cluster_id, all_images=None, cluster=None):
        if cluster_id not in self.scores:
            if all_images is None:
                raise Exception("Cannot score cluster without a valid all_images struct")
            self.scores[cluster_id]=self.calculate_score(all_images, cluster)
            self.all_results[cluster_id]=self.llm_obj.all_results
        return self.scores[cluster_id]
    
    def serialize(self):
        filter_state=super().serialize()
        filter_state['all_results']=self.all_results
        return filter_state
    
    def loader(self, json_dict):
        super().loader(json_dict)
        self.all_results=json_dict['all_results']

#################
# Functions for using the filters
#   get_filter_by_name: instantiates the right type of filter with the right arguments
#   score_all_clusters: executes multiple filter types on a single tgt dir and prompt
#################     
def get_filter_by_name(filter):
    if filter=='prob_mean_filter':
        return prob_mean_filter()
    elif filter=='pct_valid_filter':
        return pct_valid_filter()
    elif filter=='render_and_after_filter':
        return before_and_after_filter(use_render=True)
    elif filter=='before_and_after_filter':
        global fList_base, params
        return before_and_after_filter(fList_base=fList_base, params=params, use_render=False)     
    elif filter=='combo_pmf_pctV':
        F1=prob_mean_filter()
        F2=pct_valid_filter()
        return combo_filter(F1,F2)     
    elif filter=='combo_pctV_baF':
        F1=pct_valid_filter()
        F2=before_and_after_filter(use_render=False)
        return combo_filter(F1,F2)     
    elif filter=='is_pickup_filter':
        return is_pickup_filter()     
    elif filter=='is_pickup_filter0.5':
        return is_pickup_filter(image_scale=0.5)     
    elif filter=='combo_pctV_pickup':
        F1=pct_valid_filter()
        F2=is_pickup_filter()
        return combo_filter(F1,F2,"lo")     
    elif filter=='pcloud_mean_prob':
        return cluster_stat_filter('mean')
    elif filter=='pcloud_max_prob':
        return cluster_stat_filter('max')
    elif filter=='pcloud_size_filter':
        return cluster_stat_filter('pcount')
    return None

def score_all_clusters(tgt_dir, query, active_filter_list:list):
    Q=query.replace(" ","_")
    cluster_files=glob.glob(os.path.join(tgt_dir,Q+"*[0-9].pkl"))    

    # Initialize the filterBank - loading from file when possible
    filterBank={}
    for filterName in active_filter_list:
        filterBank[filterName]=get_filter_by_name(filterName)
        filterBank[filterName].loadFromFile(tgt_dir, Q)

    # Score all of the clusters - storing the results locally in the filter
    #   by the ID of the cluster
    print(f"Evaluating {query}")
    for file in cluster_files:
        with open(file, 'rb') as handle:
            A=pickle.load(handle)     
        ID=file.split('_')[-1].split('.')[0]
        for filterName in active_filter_list:
            filterBank[filterName].get_score(ID,A['all_images'],A['cluster'])

    # Save the result
    for filterName in active_filter_list:
        filterBank[filterName].save2file(tgt_dir, Q)

    return filterBank

def setup_before_and_after_filter(nerfacto_dir, 
                                  color_dir="nerf_colmap/images", 
                                  colmap_dir="nerf_colmap/colmap/sparse/0", 
                                  frame_keyword=None):
    # Need to build information from the baseline run 
    #   Specifically we need the fList_base and params variables to be created globally
    global fList_base, params

    from colmap_utils import get_camera_params, build_file_list
    initial_dir=nerfacto_dir.split('outputs')[0]
    initial_colmap_dir=initial_dir+colmap_dir
    global fList_base, params
    params=get_camera_params(colmap_dir,args.nerfacto_dir)
    fList_base=build_file_list(initial_dir+color_dir,initial_dir+"nerf_colmap/depth",initial_dir,initial_colmap_dir,frame_keyword)
    if len(fList_base.keys())==0:
        print("No images found in the base directory - check your frame keyword?")
        raise(Exception("fList_base empty"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('tgt_dir', type=str, help='Directory to evaluate')
    parser.add_argument('--queries', type=str, nargs='*', help="List of queries to evaluate")
    parser.add_argument('--filters', type=str, nargs='*', default=["prob_mean_filter", "pct_valid_filter", "pcloud_mean_prob", "pcloud_size_filter"],
                help='Set of target filters to evaluate and save out to json')
    ## These parameters are all for the before and after filter
    parser.add_argument('--nerfacto_dir',type=str,default=None,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json. Only necessary if using the before_and_after_filter')
    parser.add_argument('--color_dir',type=str,default='nerf_colmap/images',help='where are the color images of the base directory? (default=nerf_colmap/images) Only necessary if using the before_and_after_filter')
    parser.add_argument('--colmap_dir',type=str,default='nerf_colmap/colmap/sparse_geo/0',help='where are the images + cameras.txt files? (default=nerf_colmap/colmap/sparse_geo/0) Only necessary if using the before_and_after_filter')
    parser.add_argument('--frame_keyword',type=str,default="frame",help='a keyword to use when parsing the transforms file (default=frame)')    
    args = parser.parse_args()

    if 'before_and_after_filter' in args.filters:
        setup_before_and_after_filter(args.nerfacto_dir, args.color_dir, args.colmapdir, args.frame_keyword)

    for query in args.queries:
        fBank=score_all_clusters(args.tgt_dir, query, args.filters)
        ids = list(fBank[args.filters[0]].scores.keys())
        print("Filter".ljust(10), end="")
        for i in ids:
            print(f"{i}".ljust(10), end="")
        print()

        # Print each row
        for filter_name, values in fBank.items():
            print(filter_name.ljust(10), end=" ")
            for i in ids:
                out_val=values.scores[i]
                print(f"{out_val:.3f}".ljust(10), end=" ")
            print()


