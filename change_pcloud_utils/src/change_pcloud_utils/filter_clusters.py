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

def gaussian_dist_function(val1, std1):
    return np.exp(-0.5*np.power(val1/std1,2))

def label_clusters(cluster_dict, query, max_images=21):
    labels = {}

    # Create one persistent figure
    plt.ion()
    fig = plt.figure(figsize=(15, 9))

    for cluster, data in cluster_dict.items():
        
        # Select up to `max_images` at random (no replacement). Keep original list
        # if it has <= max_images entries.
        imgs = list(data.get('images', []))
        if len(imgs) > max_images:
            images = random.sample(imgs, max_images)
        else:
            images = imgs

        if not images:
            print(f"Cluster {cluster}: No images available.")
            continue
        else:
            n = len(images)
            cols = min(7, n)
            rows = (n + cols - 1) // cols

            fig.clf()  # clear old contents

            for i, img_path in enumerate(images):
                ax = fig.add_subplot(rows, cols, i+1)
                img = mpimg.imread(img_path)
                ax.imshow(img)

                # Label = everything after query_
                fname = os.path.basename(img_path).replace(".png", "")
                label = fname.replace(f"{query}_", "")
                ax.set_title(label, fontsize=10)
                ax.axis("off")

            fig.suptitle(f"{query} (Cluster {cluster})", fontsize=14)
            fig.tight_layout()
            fig.canvas.draw()  # redraw in same window

        # Ask user for labels
        valid_obj = input(f"{query} {cluster}: Is this a valid cluster? (0: valid, 1: invalid, 2: duplicate): ")
        changed = input(f"{query} {cluster}: Has the object changed? (y/n): ").strip().lower() == "y"
        pickup = input(f"{query} {cluster}: Could the object be picked up by a robot? (y/n): ").strip().lower() == "y"

        labels[cluster] = {"changed": changed, "pickup": pickup, 'valid_obj': valid_obj}

    plt.ioff()  # turn off interactive mode when done
    return labels

def build_cluster_dict(base_dir, query):
    """
    Build dict of clusters for given query.
    Each cluster has:
      'pkl': path to its .pkl file (or None if missing)
      'images': list of .png paths belonging to that cluster
    """
    cluster_dict = {}

    # Find all pkl files for this query
    pkl_files = glob.glob(os.path.join(base_dir, f"{query}_*.pkl"))
    for pkl in pkl_files:
        fname = os.path.basename(pkl)
        # "{query}_{cluster}.pkl"
        cluster = fname.replace(f"{query}_", "").replace(".pkl", "")
        cluster_dict[cluster] = {'pkl': pkl, 'images': glob.glob(os.path.join(base_dir, f"{query}_{cluster}*.png"))}

    label_file=os.path.join(base_dir,f"{query}.labels.json")
    labels = None
    if len(cluster_dict.keys())==0:
        labels={}
    else:
        try:
            if os.path.exists(label_file):
                with open(label_file,'r') as fin:
                    labels=json.load(fin)
        except Exception as e:
            print("Labels not loaded")
        if not labels:
            labels=label_clusters(cluster_dict, query)
            with open(label_file,'w') as fout:
                json.dump(labels,fout)
    return cluster_dict, labels

def get_values_from_dict(d_in, tgt_key, default_val):
    p_array=[]
    for key in d_in:
        if tgt_key in d_in[key]:
            p_array.append(d_in[key][tgt_key])
        else:
            p_array.append(default_val)
    return p_array

class evaluate():
    def __init__(self,threshold_range):
        self.records=dict()
        self.latest_record=None
        self.count_invalid=0
        self.count_valid=0
        self.threshold=threshold_range
        self.build_stat_structs(self.threshold.shape[0])
    
    def build_stat_structs(self, length):
        self.stats = {'TP': np.zeros((length,),dtype=int),
                      'FP': np.zeros((length,),dtype=int),
                      'TN': np.zeros((length,),dtype=int),
                      'FN': np.zeros((length,),dtype=int)}
    
    def load_cluster_pkl(self, cluster_fileName):
        try:
            with open(cluster_fileName, 'rb') as handle:
                cluster=pickle.load(handle)
        except Exception as e:
            print(f"Error loading {cluster_fileName} - {e}")
            return None
        return cluster
    
    def recall(self):
        result = np.true_divide(self.stats['TP'],(self.stats['TP']+self.stats['FN']))
        result[~np.isfinite(result)] = 0
        return result

    def precision(self):
        result = np.true_divide(self.stats['TP'],(self.stats['TP']+self.stats['FP']))
        result[~np.isfinite(result)] = 0
        return result

    def F_score(self):        
        R=self.recall()
        P=self.precision()
        result = np.true_divide(2*R*P,R+P)
        result[~np.isfinite(result)] = 0
        return result
    
    def add_stats(self, B):
        assert(self.stats['TP'].shape==B.stats['TP'].shape)
        self.stats['TP']+=B.stats['TP']
        self.stats['FP']+=B.stats['FP']
        self.stats['TN']+=B.stats['TN']
        self.stats['FN']+=B.stats['FN']
        self.count_invalid+=B.count_invalid
        self.count_valid+=B.count_valid

    def print_stats(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            R=self.recall()
            P=self.precision()
            F1=self.F_score()
            whichF=np.argmax(F1)
            print(f"Pct Valid Clusters: {self.count_valid/(self.count_invalid+self.count_valid)}")
            print(f"Initial F-score: {F1[0]}, P/R={P[0]}/{R[0]} ")
            print(f"Max F-score: {F1[whichF]}, P/R={P[whichF]}/{R[whichF]} ")

    def process_record(self, c_dict, labels):
        for cl_key in c_dict.keys():
            if cl_key not in labels:
                continue
            print(f"Cluster {cl_key}")
            cluster=self.load_cluster_pkl(c_dict[cl_key]['pkl'])
            v_pct=self.calculate_score(cluster['all_images'],cluster['cluster'],cluster['exp_params'])
            if labels[cl_key]['valid_obj']=="1":
                self.count_invalid+=1
            else:
                self.count_valid+=1
            if labels[cl_key]['changed']:
                self.stats['TP']+=self.threshold<=v_pct
                self.stats['FN']+=self.threshold>v_pct
            else:
                self.stats['FP']+=self.threshold<=v_pct
                self.stats['TN']+=self.threshold>v_pct    
#################
# prob_mean_filter
#   Tracks the max probability of the object per image that should be looking at the cluster
#   It uses a default detection threshold for images with no detections
#################
class prob_mean_filter(evaluate):
    def __init__(self, threshold_range=np.arange(0.1,1.0,0.01), detection_threshold=0.1):
        self.detection_threshold=detection_threshold
        super().__init__(threshold_range)
    
    def calculate_score(self, all_images, cluster=None, exp_params=None):
        pn = np.array(get_values_from_dict(all_images, 'max_prob',self.detection_threshold))
        return pn.mean() 

#################
# pct_valid_filter
#   Tracks the percentage of images with points inside the bounding box
#   to those with no detections. 
#################
class pct_valid_filter(evaluate):
    def __init__(self, threshold_range=np.arange(0.0,1.0,0.02)):
        super().__init__(threshold_range)
    
    def calculate_score(self,all_images, cluster=None, exp_params=None):
        valid_images=np.array([ 'new' in all_images[key] for key in all_images ])
        return (valid_images.sum()/len(all_images.keys()))    

#################
# pcloud_size_filter
#   Counts the number of points between all images included in the cloud. 
#################
class pcloud_size_filter(evaluate):
    def __init__(self, threshold_range=np.arange(0.0,1000,10)):
        super().__init__(threshold_range)
    
    def calculate_score(self, all_images, cluster=None, exp_params=None):
        p_count=np.array(get_values_from_dict(all_images, 'pt_count',0)).sum()
        return p_count

#################
# image cnt filter
#   Counts the number of images of the target
#################
class image_cnt_filter(evaluate):
    def __init__(self, threshold_range=np.arange(0.0,10,1)):
        super().__init__(threshold_range)
    
    def calculate_score(self, all_images, cluster=None, exp_params=None):
        valid_images=np.array([ 'new' in all_images[key] for key in all_images ])
        return valid_images.shape()
    
#################
# prob mean with size filter
#################
class prob_mean_and_size(evaluate):
    def __init__(self, threshold_range=np.arange(0.0,1.0,0.02)):
        super().__init__(threshold_range)
    
    def calculate_score(self, all_images, cluster=None, exp_params=None):
        arr=[]
        for key in all_images.keys():
            if 'pt_count' in all_images[key] and all_images[key]['pt_count']>100:
                arr.append(all_images[key]['max_prob'])
            else:
                arr.append(0.0)
        return np.array(arr).mean()

#################
# multiply filter
#################
class multiply2_filter(evaluate):
    def __init__(self, filter1:evaluate, filter2: evaluate,
                 threshold_range=np.arange(0.0,1.0,0.02)):
        self.filter1=filter1
        self.filter2=filter2
        super().__init__(threshold_range)
    
    def calculate_score(self, all_images, cluster=None, exp_params=None):
        return self.filter1.calculate_score(all_images, cluster, exp_params)*self.filter2.calculate_score(all_images, cluster, exp_params)

#################
# LLM - Before And After Filter
#   Tracks the percentage of images with points inside the bounding box
#   to those with no detections. 
#################
class before_and_after_filter(evaluate):
    def __init__(self, threshold_range=np.arange(0.0,1.0,0.02), use_render=True):
        self.use_render=use_render
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
        super().__init__(threshold_range)
    
    def set_closest_images(self, all_images, cluster_centroid, fList_new):
        global params, fList_base
        # find images in the baseline dataset that can see the target cluster
        rel_imgs=identify_related_images_global_pose(params, fList_base, cluster_centroid, None, 0.5)
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
                M=np.matmul(params.rot_matrix,fList_new.get_pose(int(key)))
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
    
    def calculate_score(self,all_images, cluster=None, exp_params=None):
        if not self.use_render: # need to add "before" images to all_images
            all_images=self.set_closest_images(all_images,cluster.centroid,exp_params['fList_new'])
        # Need to flip dimensions of the images - they are inverted
        for key in all_images.keys():
            if 'new' in all_images[key]:
                all_images[key]['new']=all_images[key]['new'][:,:,[2,1,0]]
            if 'render' in all_images[key]:
                all_images[key]['render']=all_images[key]['render'][:,:,[2,1,0]]
        pickup_pct, new_pct=self.ba_cluster_filt.evaluate_multiple_image_pairs(all_images, before_key=self.before_key)
        return new_pct
    
#################
# is_pickup_filter - query llm about picking up objects specifically
#################
class is_pickup_filter(evaluate):
    def __init__(self, threshold_range=np.arange(0.0,1.0,0.02),image_scale=1.0):
        self.llm_obj=multi_view_cluster_filter()
        super().__init__(threshold_range)
    
    def calculate_score(self, all_images, cluster=None, exp_params=None):
        return self.llm_obj.evaluate_cluster(all_images)

def get_filter_by_name(filter):
    if filter=='prob_mean_filter':
        return prob_mean_filter()
    elif filter=='pct_valid_filter':
        return pct_valid_filter()
    elif filter=='render_and_after_filter':
        return before_and_after_filter(use_render=True)
    elif filter=='before_and_after_filter':
        return before_and_after_filter(use_render=False)     
    elif filter=='pcloud_size_filter':
         return pcloud_size_filter()     
    elif filter=='prob_mean_and_size':
        return prob_mean_and_size()   
    elif filter=='image_cnt_filter':
        return image_cnt_filter()  
    elif filter=='combo_pmf_pctV':
        F1=prob_mean_filter()
        F2=pct_valid_filter()
        return multiply2_filter(F1,F2)     
    elif filter=='combo_pctV_baF':
        F1=pct_valid_filter()
        F2=before_and_after_filter(use_render=False)
        return multiply2_filter(F1,F2)     
    elif filter=='is_pickup_filter':
        return is_pickup_filter()     
    elif filter=='combo_pctV_pickup':
        F1=pct_valid_filter()
        F2=is_pickup_filter()
        return multiply2_filter(F1,F2)     
    return None

###########################
## multi-evaluator
##   this is a hack - stores the intermediate evaluations per directory
##   in a file so that we don't have to re-run laborious LLM calls repeatedly
##   Designed to be generic enough that it will work with arbitrary evaluation
##   functions
###########################
def multi_evaluator(tgt_dir, queries):
    # active=['prob_mean_filter','pct_valid_filter','combo_pmf_pctV']
    active=['pct_valid_filter']
    # active=['pct_valid_filter','combo_pmf_pctV','before_and_after_filter','combo_pctV_baF']
    # active=['is_pickup_filter', 'combo_pctV_pickup']
    filterBank={}
    for filter in active:
        filterName=f"{tgt_dir}/{filter}.filt"
        try:
            # Try to open this file - if exists, move on, else will need to execute
            with open(filterName, 'rb') as handle:
                filterBank[filter]=pickle.load(handle)
            is_valid=[ prompt in filterBank[filter] for prompt in queries ]
            if sum(is_valid)<len(queries):
                filterBank[filter]=None
                raise Exception("all prompts not present - rebuilding {filterName}")
        except Exception as e:
            # When a new filter is added, put the initialization code here - keeping things separated by prompt
            #    for further analysis
            filterBank[filter]={ prompt:get_filter_by_name(filter) for prompt in queries }

            # Execution code
            for prompt in queries:
                print(f"Evaluating {prompt}")
                c_dict,labels=build_cluster_dict(tgtD, prompt.replace(' ','_'))
                try:                    
                    filterBank[filter][prompt].process_record(c_dict,labels)
                except Exception as e:
                    pdb.set_trace()
            
            # Save the result
            with open(filterName,'wb') as handle:
                pickle.dump(filterBank[filter], handle, protocol=pickle.HIGHEST_PROTOCOL)    
    return filterBank

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('--tgt_dir', type=str, nargs='*', default=None,help='set of directories to evaluate with the same baseline')
    parser.add_argument('--queries', type=str, nargs='*', default=["clothing", "dishes", "general clutter", "small items"],
                help='Set of target queries to build point clouds for - default is [General clutter, Small items on surfaces, Floor-level objects, Decorative and functional items, Trash items]')
    parser.add_argument('--color_dir',type=str,default='nerf_colmap/images',help='where are the color images of the base directory? (default=nerf_colmap/images)')
    parser.add_argument('--colmap_dir',type=str,default='nerf_colmap/colmap/sparse_geo/0',help='where are the images + cameras.txt files? (default=nerf_colmap/colmap/sparse_geo/0)')
    parser.add_argument('--frame_keyword',type=str,default="frame",help='a keyword to use when parsing the transforms file (default=new)')    
    args = parser.parse_args()

    from colmap_utils import get_camera_params, build_file_list
    initial_dir=args.nerfacto_dir.split('outputs')[0]
    colmap_dir=initial_dir+args.colmap_dir
    global fList_base, params
    params=get_camera_params(colmap_dir,args.nerfacto_dir)
    fList_base=build_file_list(initial_dir+args.color_dir,initial_dir+"nerf_colmap/depth",initial_dir,colmap_dir,args.frame_keyword)
    if len(fList_base.keys())==0:
        print("No images found in the base directory - check your frame keyword?")
        raise(Exception("fList_base empty"))

    # cluster_dict=dict()
    combined_dict=None
    
    # for tgtD in args.tgt_dir:
    #     for prompt in args.queries:
    #         c_dict,labels=build_cluster_dict(tgtD, prompt.replace(' ','_'))
    #         pmf_dict[prompt].process_record(c_dict,labels)
    for tgtD in args.tgt_dir:
        filterBank=multi_evaluator(tgtD, args.queries)
        if combined_dict is None:
            combined_dict=filterBank
        else:
            for key1 in filterBank.keys():
                for key2 in filterBank[key1].keys():
                    combined_dict[key1][key2].add_stats(filterBank[key1][key2])

    for key in filterBank.keys():
        eval_combo=None
        print(f"#### {key} ####")
        for prompt in args.queries:
            print(f" ****************** {prompt} ****************** ")
            combined_dict[key][prompt].print_stats()
            if eval_combo is None:
                eval_combo=evaluate(combined_dict[key][prompt].threshold)
            eval_combo.add_stats(combined_dict[key][prompt])
        
        print(f" ****************** {key} COMBINED ****************** ")
        eval_combo.print_stats()
        print("")
        print("")