import numpy as np
from sklearn.cluster import DBSCAN
from copy import copy
from grid import grid3D, max_grid3D
import pdb

def check_single_dim_overlap(a_min, a_max, b_min, b_max):
    if a_min<b_min:
        if a_max>b_min:
            return True
    elif a_max>b_min and a_min<b_max:
        return True
    return False

class cluster():
    def __init__(self, pts:np.ndarray, scores:np.ndarray, save_raw:bool=False):
        self.mean_=[pts.mean(0)]
        self.min_=[pts.min(0)]
        self.max_=[pts.max(0)]
        self.scores_mean_=[scores.mean()]
        self.scores_max_=[scores.max()]
        self.count_=[pts.shape[0]]
        if save_raw:
            self.pts_=[pts]
            self.scores_=[scores]
        else:
            self.pts_=[np.zeros((0,3),dtype=float)]
            self.scores_=[np.zeros((0,1),dtype=float)]

    def add_cluster(self, new_cl):
        self.mean_+=new_cl.mean_
        self.max_+=new_cl.max_
        self.min_+=new_cl.min_
        self.scores_mean_+=new_cl.scores_mean_
        self.scores_max_+=new_cl.scores_max_
        self.pts_+=new_cl.pts_
        self.scores_+=new_cl.scores_

    def is_pt_in_cluster(self, xyz):
        for idx in range(self.count_clusters()):
            value=(xyz>=self.min_[idx])*(xyz<self.max_[idx])
            if value.prod()==1.0:
                return True
        return False

    def mean(self):
        if len(self.mean_)==1:
            return self.mean_[0]
        return np.array(self.mean_,dtype=float).mean(0)
    
    def min(self):
        if len(self.min_)==1:
            return self.min_[0]
        return np.array(self.min_,dtype=float).min(0)

    def max(self):
        if len(self.max_)==1:
            return self.max_[0]
        return np.array(self.max_,dtype=float).max(0)
    
    def score_mean(self):
        if len(self.scores_mean_)==1:
            return self.scores_mean_[0]
        return np.array(self.scores_mean_,dtype=float).mean()

    def score_max(self):
        if len(self.scores_max_)==1:
            return self.scores_max_[0]
        return np.array(self.scores_max_,dtype=float).max()

    def count(self):
        if len(self.count_)==1:
            return self.count_[0]
        return np.array(self.count_,dtype=float).sum()

    def is_overlap(self, new_cl):
        for a_idx in range(len(self.mean_)):
            for b_idx in range(len(new_cl.mean_)):        
                value=True
                for xyz in range(3):
                    value = value and check_single_dim_overlap(self.min_[a_idx][xyz], self.max_[a_idx][xyz], new_cl.min_[b_idx][xyz], new_cl.max_[b_idx][xyz])
                if value:
                    return True

    def count_clusters(self):
        return len(self.mean_)
        
class cluster_history():
    def __init__(self):
        self.raw_clusters={} # raw clusters collected per image
        self.agg_clusters={} # agglomerated clusters across a mission
    
    def add_clusters(self, detection_target:str, cluster_list:list, image_key:str):
        if detection_target not in self.raw_clusters:
            self.setup_category(detection_target)
        self.raw_clusters[detection_target]['clusters'].append(cluster_list)
        self.raw_clusters[detection_target]['images'].append(image_key)

    def setup_category(self, detection_target):
        self.raw_clusters[detection_target]={'clusters': [], 'images': []}

    def count_clusters(self, detection_target):
        if detection_target not in self.raw_clusters:
            return None
        count = 0
        for cluster_list in self.raw_clusters[detection_target]['clusters']:
            count+=len(cluster_list)
        return count

    def count_images(self, detection_target):
        if detection_target not in self.raw_clusters:
            return None
        return len(self.raw_clusters[detection_target]['images'])

    def all_cluster_sizes(self, detection_target):
        if detection_target not in self.raw_clusters:
            return None
        counts=[]
        for cluster_list in self.raw_clusters[detection_target]['clusters']:
            for cl_ in cluster_list:
                counts.append(cl_.count())
        return np.array(counts)

    def all_cluster_scores_max(self, detection_target):
        if detection_target not in self.raw_clusters:
            return None
        scores=[]
        for cluster_list in self.raw_clusters[detection_target]['clusters']:
            for cl_ in cluster_list:
                scores.append(cl_.score_max())
        return np.array(scores)

    def all_cluster_scores_mean(self, detection_target):
        if detection_target not in self.raw_clusters:
            return None
        scores=[]
        for cluster_list in self.raw_clusters[detection_target]['clusters']:
            for cl_ in cluster_list:
                scores.append(cl_.score_mean())
        return np.array(scores)

    def visualize_agg_clusters(self, detection_target):
        if detection_target not in self.agg_clusters:
            return None
        # Get boundaries
        if len(self.agg_clusters[detection_target])==0:
            print("No agglomerative clusters to visualize")
            return None
        
        pts=np.zeros((0,3),dtype=float)
        for cl_key in self.agg_clusters[detection_target]:
            pts=np.vstack((pts,
                           self.agg_clusters[detection_target][cl_key].min(),
                           self.agg_clusters[detection_target][cl_key].max()))
        num_cells=((pts.max(0)-pts.min(0)+0.1)/0.05).astype(int)
        vis_grid=max_grid3D(pts.min(0),pts.max(0),num_cells)
        vis_grid.create_empty_grid()
        for cl_key in self.agg_clusters[detection_target]:
            cl_=self.agg_clusters[detection_target][cl_key]
            min_xyz=cl_.min()
            max_xyz=cl_.max()
            min_cell=vis_grid.get_cell(min_xyz[0],min_xyz[1],min_xyz[2])
            max_cell=vis_grid.get_cell(max_xyz[0],max_xyz[1],max_xyz[2])
            # Now go through each cell in that range and check for 
            #   being in the cluster - still imprecise because we are using 
            #   boxes in the raw clusters, but better than blocking out a super huge
            #   prizm
            for cX in range(min_cell[0],max_cell[0]+1):
                for cY in range(min_cell[1],max_cell[1]+1):
                    for cZ in range(min_cell[2],max_cell[2]+1):
                        if vis_grid.is_cell_in_bounds(cX,cY,cZ):
                            xyz=vis_grid.get_xyz(cX,cY,cZ)
                            if cl_.is_pt_in_cluster(xyz):
                                vis_grid.set_cell_value(cX,cY,cZ,cl_.count_clusters())
        return vis_grid

    def build_agglomerative_clusters(self, detection_target, min_agg_count=1):
        cluster_list=dict()
        count = 0
        for clist_ in self.raw_clusters[detection_target]['clusters']:
            for cl_ in clist_:
                cluster_list[count]=copy(cl_)
                count+=1
        
        # all_min = [ cluster_list[key].min() for key in cluster_list ]
        # all_max = [ cluster_list[key].max() for key in cluster_list ]
        change = 1
        count=0
        while (change>0) and (count<100):
            change = 0
            count+=1
            all_keys = [key for key in cluster_list.keys()]   
            for key1 in all_keys:
                # key might have been removed already - need to check first
                if key1 not in cluster_list:
                    continue
                # Otherwise, check for overlap against all existing clusters
                merge_candidates=[]
                for key2 in cluster_list.keys():
                    if key1==key2:
                        continue
                    if cluster_list[key1].is_overlap(cluster_list[key2]):
                        merge_candidates.append(key2)
                # And now merge
                for key2 in merge_candidates:
                    change+=1
                    cluster_list[key1].add_cluster(cluster_list[key2])
                    cluster_list.pop(key2)

        # Remove clusters that do not have any corroborating evidence
        bad_clusters=[]
        for key in cluster_list:
            if cluster_list[key].count_clusters()<min_agg_count:
                bad_clusters.append(key)
        for key in bad_clusters:
            cluster_list.pop(key)
        self.agg_clusters[detection_target]=cluster_list
    
    def overlap_with_agg_cluster(self, detection_target, new_cluster:cluster):
        for key in self.agg_clusters[detection_target]:
            if self.agg_clusters[detection_target][key].is_overlap(new_cluster):
                return key
        return None


