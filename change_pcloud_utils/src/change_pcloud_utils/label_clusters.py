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
from change_pcloud_utils.filter_clusters import cluster_filter, get_filter_by_name, score_all_clusters

def gaussian_dist_function(val1, std1):
    return np.exp(-0.5*np.power(val1/std1,2))

# Manually label clusters - need to identify 
#   1) Has the object changed? 
#   2) Could it be picked up by the robot?
#   3) If it has changed, what kind of object is it? ... to be used in determining duplicates
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
        changed = input(f"{query} {cluster}: Has the object changed? (y/n): ").strip().lower() == "y"
        pickup = input(f"{query} {cluster}: Could the object be picked up by a robot? (y/n): ").strip().lower() == "y"
        if changed:
            name = input(f"{query} {cluster}: Name the changed object ").strip().lower()
        else:
            name="unchanged"

        labels[cluster] = {"changed": changed, "pickup": pickup, 'name': name}

    plt.ioff()  # turn off interactive mode when done
    return labels

def build_cluster_label_dict(base_dir, query, suffix="labels.json"):
    """
    Build dict of cluster labels for given query.
    """
    pkl_files = glob.glob(os.path.join(base_dir, f"{query}_*.pkl"))

    # Step 1 - try to load labels from an existing file
    label_file=os.path.join(base_dir,f"{query}.{suffix}")
    labels = None
    try:
        if os.path.exists(label_file):
            with open(label_file,'r') as fin:
                labels=json.load(fin)
            # if len(labels.keys())==len(pkl_files):
            #     print(f"Labels loaded: {base_dir} - {query}")
            return labels
    except Exception as e:
        print(f"Labels NOT loaded: {base_dir} - {query}")

    # Step 2 - loading failed ... need to build cluster dict and
    #           go through manual labeling process
    cluster_dict = {}
    for pkl in pkl_files:
        fname = os.path.basename(pkl)
        # "{query}_{cluster}.pkl"
        ID=int(pkl.split('_')[-1].split('.')[0])
        cluster = fname.replace(f"{query}_", "").replace(".pkl", "")
        cluster_dict[cluster] = {'pkl': pkl, 
                                 'images': glob.glob(os.path.join(base_dir, f"{query}_{cluster}*.png"))}

    # Step 3 - call the label clusters routine and save the result
    if len(cluster_dict.keys()):
        labels=label_clusters(cluster_dict, query)
    else:
        labels=dict()
    
    with open(label_file,'w') as fout:
        json.dump(labels,fout)

    return labels

def get_values_from_dict(d_in, tgt_key, default_val):
    p_array=[]
    for key in d_in:
        if tgt_key in d_in[key]:
            p_array.append(d_in[key][tgt_key])
        else:
            p_array.append(default_val)
    return p_array

# The evaluate function processes existing filters
#   where all the clusters have already been scored
#   and the results stored in the filter. It stores
#   data internally - retrieve after each call to 
#   process_cluster
class evaluate():
    def __init__(self, threshold_range:np.ndarray):
        self.records=dict()
        self.latest_record=None
        self.count_invalid=0
        self.count_valid=0
        self.filter=filter
        self.rebuild_stat_structs(threshold_range)
    
    def rebuild_stat_structs(self, threshold_range):
        self.threshold=threshold_range
        length=self.threshold.shape[0]
        self.tp_object_list={val:[] for val in range(len(self.threshold))}
        self.fn_object_list={val:[] for val in range(len(self.threshold))}
        self.stats={'TP': np.zeros((length,),dtype=int),
                    'FP': np.zeros((length,),dtype=int),
                    'TN': np.zeros((length,),dtype=int),
                    'FN': np.zeros((length,),dtype=int)}
    
    def recall(self):
        result = np.true_divide(self.stats['TP'],(self.stats['TP']+self.stats['FN']))
        result[~np.isfinite(result)] = 0
        return result

    def precision(self):
        result = np.true_divide(self.stats['TP'],(self.stats['TP']+self.stats['FP']))
        result[~np.isfinite(result)] = 0
        return result

    def specificity(self):
        result = np.true_divide(self.stats['TN'],(self.stats['TN']+self.stats['FP']))
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
            S=self.specificity()
            F1=self.F_score()
            count_objects=[ len(np.unique(self.tp_object_list[idx])) for idx in self.tp_object_list ]
            whichF=np.argmax(F1)
            print(f"Pct Valid Clusters: {self.count_valid/(self.count_invalid+self.count_valid)}")
            print(f"Initial F-score: {F1[0]}, P/R={P[0]}/{R[0]} ")
            print(f"Max F-score: {F1[whichF]}, P/R={P[whichF]}/{R[whichF]} ")

            print("Threshold",end=", ")
            for idx in range(len(self.threshold)):
                print(self.threshold[idx],end=", ")
            print("")
            print("Precision",end=", ")
            for idx in range(len(self.threshold)):
                print(P[idx],end=", ")
            print("")
            print("Specificity",end=", ")
            for idx in range(len(self.threshold)):
                print(S[idx],end=", ")
            print("")
            print("Object Count",end=", ")
            for idx in range(len(self.threshold)):
                print(count_objects[idx],end=", ")
            print("")

    def process_filter(self, filter:cluster_filter, labels:dict, dir_label:str=''):        
        for key in filter.scores.keys():
            if key in labels:   # ignore clusters that did not generate any images to be labeled
                self.count_valid+=1
                v_pct=filter.scores[key]
                if labels[key]['changed']:
                    pos=self.threshold<=v_pct
                    neg=self.threshold>v_pct
                    whichP=np.where(pos)[0]
                    whichN=np.where(neg)[0]
                    
                    # Check for multiple objects in the label
                    for itm in labels[key]['name'].split(','):
                        if itm[-1]==' ':    #remove blank spaces
                            itm=itm[:-1]
                        if itm[0]==' ':
                            itm=itm[1:]
                        # Add to the lists
                        for pp in whichP:
                            self.tp_object_list[pp].append(f"{dir_label}-{labels[key]['name']}")
                        for nn in whichN:
                            self.fn_object_list[nn].append(f"{dir_label}-{labels[key]['name']}")

                    self.stats['TP']+=pos
                    self.stats['FN']+=self.threshold>v_pct
                else:
                    self.stats['FP']+=self.threshold<=v_pct

                    self.stats['TN']+=self.threshold>v_pct    
            else:
                self.count_invalid+=1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_dir', type=str, nargs='*', default=None,help='set of directories to evaluate with the same baseline')
    parser.add_argument('--queries', type=str, nargs='*', default=["clothing", "dishes", "general clutter", "small items"],
                help='Set of target queries to build point clouds for - default is [General clutter, Small items on surfaces, Floor-level objects, Decorative and functional items, Trash items]')
    parser.add_argument('--filters', type=str, nargs='*', default=["prob_mean_filter", "pct_valid_filter", "pcloud_mean_prob"],
                help='Set of target filters to evaluate and save out to json')    
    parser.add_argument('--label_suffix',type=str, default="labels.json", help="ending of the label file")
    args = parser.parse_args()


    # Create the evaluation functions
    all_results={filterName:evaluate(np.arange(0,1.0,0.01)) for filterName in args.filters}

    # Run the evaluation
    for tgt_dir in args.tgt_dir:
        print(tgt_dir)
        for query in args.queries:
            Q=query.replace(" ","_")
            # Get the labels - 
            labels=build_cluster_label_dict(tgt_dir, Q, args.label_suffix)

            # Load the filters
            filterBank=score_all_clusters(tgt_dir,Q,args.filters)

            # Save to the evaluation function
            for filterName in args.filters:
                all_results[filterName].process_filter(filterBank[filterName], labels, tgt_dir)

    for key in all_results.keys():
        print(f" ****************** {key} **************** ")
        all_results[key].print_stats()

    # for key in filterBank.keys():
    #     eval_combo=None
    #     print(f"#### {key} ####")
    #     for prompt in args.queries:
    #         print(f" ****************** {prompt} ****************** ")
    #         combined_dict[key][prompt].print_stats()
    #         if eval_combo is None:
    #             eval_combo=evaluate(combined_dict[key][prompt].threshold)
    #         eval_combo.add_stats(combined_dict[key][prompt])
        
    # print(f" ****************** {key} COMBINED ****************** ")
    # eval_combo.print_stats()
    # print("")
    # print("")