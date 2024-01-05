import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import pdb
from grid import evidence_grid3D

def check_single_dim_overlap(a_min, a_max, b_min, b_max):
    if a_min<b_min:
        if a_max>b_min:
            return True
    elif a_max>b_min and a_min<b_max:
        return True
    return False

class obj_cluster():
    def __init__(self, mean_, max_, min_, value):
        self.mean=[mean_]
        self.max=[max_]
        self.min=[min_]
        self.value=[value]
    
    def add_cluster(self, new_cl):
        self.mean+=new_cl.mean
        self.max+=new_cl.max
        self.min+=new_cl.min
        self.value+=new_cl.value

    def is_overlap(self, new_cl):
        for a_idx in range(len(self.mean)):
            for b_idx in range(len(new_cl.mean)):        
                value=True
                for xyz in range(3):
                    value = value and check_single_dim_overlap(self.min[a_idx][xyz], self.max[a_idx][xyz], new_cl.min[b_idx][xyz], new_cl.max[b_idx][xyz])
                if value:
                    return True

    def total_value(self, is_log_odds=True):
        total=0
        for val in self.value:
            if is_log_odds:
                total+=np.log(val+1e-6)-np.log(1-val)
            else:
                total+=val
        return total
    
    def count(self):
        return len(self.mean)

# F1="/data2/datasets/office/no_person/initial/depth_clusters.stats.pkl"
# F1="/data2/datasets/office/no_person/monitor/rotated/depth_clusters.stats.pkl"
# F1="/data2/datasets/office/no_person/unchanged/rotated/depth_clusters.stats.pkl"
def build_cluster_list(all_stats, num_files=100):
    all_keys=[key for key in all_stats.keys()]
    np.random.shuffle(all_keys)
    all_clusters=dict()
    for key in all_keys[:num_files]:
        print(key)
        globalPose=all_stats[key]['pose']
        minXYZ=np.ones((4,))
        maxXYZ=np.ones((4,))
        meanXYZ=np.ones((4,))

        for s_idx, stats in enumerate(all_stats[key]['stats']):
            if s_idx not in all_clusters:
                all_clusters[s_idx]={'mean': [], 'max': [], 'min': [], 'p_mean': [], 'p_max': []}

            for cl_ in stats:
                minXYZ[:3]=cl_['min']
                maxXYZ[:3]=cl_['max']
                meanXYZ[:3]=cl_['mean']
                gMinXYZ=np.matmul(globalPose,minXYZ)
                gMaxXYZ=np.matmul(globalPose,maxXYZ)
                gMeanXYZ=np.matmul(globalPose,meanXYZ)
                bounds=np.array([gMinXYZ[:3],gMaxXYZ[:3]])
                all_clusters[s_idx]['mean'].append(gMeanXYZ[:3])
                all_clusters[s_idx]['min'].append(bounds.min(0))
                all_clusters[s_idx]['max'].append(bounds.max(0))
                all_clusters[s_idx]['p_mean'].append(cl_['p_mean'])
                all_clusters[s_idx]['p_max'].append(cl_['p_max'])
    return all_clusters

def agglomerative_cluster(cluster_dict):
    cluster_list=dict()
    for idx in range(len(cluster_dict['mean'])):
        cluster_list[idx]=obj_cluster(cluster_dict['mean'][idx], cluster_dict['max'][idx], cluster_dict['min'][idx], cluster_dict['p_max'][idx])
    
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
        print("Iteration %d: %d changes"%(count, change))
    size_arr=[]
    key_arr=[]
    for key in cluster_list.keys():
        print("%d) %d clusters, %0.3f"%(key, len(cluster_list[key].mean),cluster_list[key].total_value()))
        size_arr.append(len(cluster_list[key].mean))
        key_arr.append(key)
    pdb.set_trace()
    return cluster_list

def get_all_clusters(all_stats, num_files=100):
    egrid=None
    all_keys=[key for key in all_stats.keys()]
    np.random.shuffle(all_keys)
    for key in all_keys[:num_files]:
        print(key)
        if egrid is None:     
            egrid=evidence_grid3D([-1.0, -5.0, -0.3],[5.0, 0.5, 2.0],[60,55,23],len(all_stats[key]['stats']))
        globalPose=all_stats[key]['pose']
        minXYZ=np.ones((4,))
        maxXYZ=np.ones((4,))
        for s_idx, stats in enumerate(all_stats[key]['stats']):
            for cl_ in stats:
                print(cl_)
                minXYZ[:3]=cl_['min']
                maxXYZ[:3]=cl_['max']
                gMinXYZ=np.matmul(globalPose,minXYZ)
                gMaxXYZ=np.matmul(globalPose,maxXYZ)
                C1=egrid.get_cell(gMinXYZ[0],gMinXYZ[1],gMinXYZ[2])
                C2=egrid.get_cell(gMaxXYZ[0],gMaxXYZ[1],gMaxXYZ[2])
                minC=np.min([C1,C2],0)
                maxC=np.max([C1,C2],0)
                for xx in np.arange(minC[0],maxC[0]+1):
                    for yy in np.arange(minC[1],maxC[1]+1):
                        for zz in np.arange(minC[2],maxC[2]+1):
                            if egrid.is_cell_in_bounds(xx,yy,zz):
                                egrid.grid[s_idx,xx,yy,zz]+=0.1
    return egrid

def get_dbscan_cluster_data(egrid, dimension):
    from sklearn.cluster import DBSCAN
    pts0=egrid.get_thresholded_points(dimension).transpose()
    CL2=DBSCAN(eps=0.2, min_samples=5).fit(pts0[:,:3],sample_weight=pts0[:,3])
    clusters=[]
    for idx in range(10):
        whichP=np.where(CL2.labels_== idx)
        if len(whichP[0])<1:
            break
        # clusters.append(np.hstack((len(whichP[0]), pts0[whichP].mean(0),pts0[whichP].std(0))))
        clusters.append(np.hstack((len(whichP[0]), pts0[whichP][:,2:].mean(0),pts0[whichP].std(0))))
    # If only one cluster, add a zero cluster to the end
    while len(clusters)<2:
        # clusters.append([0,0,0,0,0,0,0,0,0])
        clusters.append([0,0,0,0,0,0,0])
    cl_arr=np.array(clusters)
    whichCl=np.argsort(cl_arr[:,0])
    return np.ndarray.flatten(cl_arr[whichCl[[-1,-2]],:])

if __name__ == '__main__':
    F1=["/data2/datasets/office/no_person/initial/rotated/depth_clusters.stats.pkl",
        # "/data2/datasets/office/no_person/unchanged/rotated/depth_clusters.stats.pkl",
        "/data2/datasets/office/no_person/monitor/rotated/depth_clusters.stats.pkl",
        "/data2/datasets/office/no_person/bookshelf/rotated/depth_clusters.stats.pkl"]

    # egrid=dict()
    clusters=dict()
    for idx, file in enumerate(F1):
        with open(file, "rb") as fin:
            all_stats=pickle.load(fin)

        clusters[idx]=[[],[],[],[]]
        for i in range(20):
            cl_list=build_cluster_list(all_stats, num_files=200)
            for dim_ in range(4):
                agglomerative_cluster(cl_list[dim_])
            # for dim_ in range(4):
            #     clusters[idx][dim_].append(get_dbscan_cluster_data(egrid,dim_))
        pdb.set_trace()

    pdb.set_trace()

    xx,yy,zz=egrid[0].get_ranges()
    # plt.contourf(xx,yy,(egrid.grid[2]>0).sum(2).transpose())
    # plt.show()
    pdb.set_trace()
