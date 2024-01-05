# from estimate_change_by_room import weighted_kmeans
import numpy as np
import pickle
import pdb

import argparse
from accumulate_evidence import map_run
import os
import pdb
import numpy as np
import scipy
import pickle
from grid import evidence_grid3D
import matplotlib.pyplot as plt

# def kmeans(points, weights, num_means):

def get_kmeans(pts, num_means):
    RES=scipy.cluster.vq.kmeans2(pts,2)

class weighted_kmeans():
    def __init__(self, points):
        self.points=points
        self.best_solution=None
    
    def get_assignments(self, means):
        # Which mean is each point assigned too?
        dist = np.zeros((self.points.shape[0], means.shape[0]),dtype=float)
        for m_idx, mn in enumerate(means):
            dist[:,m_idx]=np.sqrt(np.power(self.points[:,:3]-mn,2).sum(1))
        return dist.argmin(1)
    
    def recalculate_means(self, num_means, assignments):
        means=np.zeros((num_means, 3),dtype=float)
        for m_idx in range(num_means):            
            whichP=self.points[(assignments==m_idx),:]
            total_weight=whichP[:,3].sum()
            for dim in range(3):
                means[m_idx, dim]=(whichP[:,dim]*whichP[:,3]).sum()/total_weight
                # if np.isnan(means[m_idx, dim]).sum()>0:
                #     pdb.set_trace()
        return means

    def weighted_error(self, means, assignments):
        err = 0
        for m_idx, mn in enumerate(means):
            whichP=self.points[assignments==m_idx]
            dist=np.sqrt(np.power(whichP[:,:3]-mn,2).sum(1))
            err+=(dist*whichP[:,3]).sum()
        return err

    def run(self, num_means, max_iterations=100):
        # Pick points to use as randomly initialized means
        whichP=np.random.choice(np.arange(self.points.shape[0]),num_means)
        means=self.points[whichP,:3]
        current_assignments=self.get_assignments(means)
        for iter_ in range(max_iterations):
            new_means=self.recalculate_means(num_means, current_assignments)
            if np.isnan(new_means).sum()>0:
                # A mean has zero assignments - use as a termination condition
                break
            new_assignments=self.get_assignments(new_means)
            if (new_assignments-current_assignments).sum()==0:
                break
            current_assignments=new_assignments
        err=self.weighted_error(means, current_assignments)
        return means, current_assignments, err

def rebuild_grid(grid):
    egrid=evidence_grid3D([-1.0, -5.0, -0.3],[5.0, 0.5, 2.0],[60,55,23],grid.shape[0])
    egrid.load_raw_grid(grid)
    return egrid

def get_grid_stats(grid):
    egrid=rebuild_grid(grid)
    pts0=egrid.get_thresholded_points(0).transpose()
    CL=weighted_kmeans(pts0)
    err = np.zeros((10,20),dtype=float)
    for num_means in range(10):
        for idx in range(20):
            err[num_means,idx]=CL.run(num_means+1)[2]
            print(err)
    
    pdb.set_trace()

# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.cluster.hierarchy import dendrogram
# from sklearn.cluster import AgglomerativeClustering

# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram

#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count

#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)

#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)

def plot_dbscan_clusters(grid, dimension):
    from sklearn.cluster import DBSCAN
    egrid=rebuild_grid(grid)
    pts0=egrid.get_thresholded_points(dimension,-1.0).transpose()
    CL2=DBSCAN(eps=0.2, min_samples=5).fit(pts0[:,:3],sample_weight=pts0[:,3])
    xx, yy, zz = egrid.get_ranges()
    plt.contourf(xx,yy,(grid[dimension]*(grid[dimension]>0)).sum(2).transpose())
    for idx in range(10):
        whichP=np.where(CL2.labels_== idx)
        if len(whichP[0])<1:
            break
        minP=pts0[whichP].min(0)
        maxP=pts0[whichP].max(0)
        X=[minP[0],minP[0],maxP[0],maxP[0],minP[0]]
        Y=[minP[1],maxP[1],maxP[1],minP[1],minP[1]]
        plt.plot(X,Y,color=[1,0,0])
    plt.show()

def get_dbscan_cluster_data(grid, dimension):
    from sklearn.cluster import DBSCAN
    egrid=rebuild_grid(grid)
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

def get_prob_stats(grid, dimension):
    egrid=rebuild_grid(grid)
    pts0=egrid.get_thresholded_points(dimension).transpose()
    return np.hstack((pts0.shape[0], pts0[:,2:].mean(0), pts0[:,2:].std(0)))

def gaussian_pdf_std(mean, stdev, V):
    return np.exp(-0.5*np.power((mean[:]-V)/stdev,2))

def gaussian_pdf_multi(mean, inverse_cov, V):
    deltaV=mean-V
    A1=np.matmul(deltaV.transpose(),inverse_cov)
    return float(np.exp(-0.5*np.matmul(A1,deltaV)))

def log_odds_summation(arr):
    return (np.log(arr)-np.log(1-arr)).sum()

files = ['']
F1="/data2/datasets/office/no_person/initial/rotated/many_grids-intermediate.pkl"
F2="/data2/datasets/office/no_person/bookshelf/rotated/many_grids-intermediate-ray.pkl"
F3="/data2/datasets/office/no_person/monitor/rotated/many_grids-intermediate-ray.pkl"
F4="/data2/datasets/office/no_person/unchanged/rotated/many_grids-intermediate.pkl"

with open(F1,"rb") as fin:
    AG1=np.array(pickle.load(fin))

with open(F2,"rb") as fin:
    AG2=np.array(pickle.load(fin))

with open(F3,"rb") as fin:
    AG3=np.array(pickle.load(fin))

with open(F4,"rb") as fin:
    AG4=np.array(pickle.load(fin))

# plot_dbscan_clusters(AG1[0],0)
dimension=2
all_cl1=[]
for idx in range(AG1.shape[0]):
    # all_cl1.append(get_prob_stats(AG1[idx],dimension))
    all_cl1.append(get_dbscan_cluster_data(AG1[idx],dimension))
all_cl2=[]
for idx in range(AG2.shape[0]):
    # all_cl2.append(get_prob_stats(AG2[idx],dimension))
    all_cl2.append(get_dbscan_cluster_data(AG2[idx],dimension))
all_cl3=[]
for idx in range(AG3.shape[0]):
    # all_cl3.append(get_prob_stats(AG3[idx],dimension))
    all_cl3.append(get_dbscan_cluster_data(AG3[idx],dimension))
all_cl4=[]
for idx in range(AG4.shape[0]):
    # all_cl4.append(get_prob_stats(AG4[idx],dimension))
    all_cl4.append(get_dbscan_cluster_data(AG4[idx],dimension))

all_cl1=np.array(all_cl1)
all_cl2=np.array(all_cl2)
all_cl3=np.array(all_cl3)
all_cl4=np.array(all_cl4)

model1={'mean': all_cl1.mean(0), 'std': all_cl1.std(0), 'cov': np.cov(all_cl1,rowvar=False)}
model2={'mean': all_cl4.mean(0), 'std': all_cl4.std(0), 'cov': np.cov(all_cl4,rowvar=False)}
model1['det']= np.linalg.det(model1['cov'])
model2['det']= np.linalg.det(model2['cov'])
model1['inv']=np.linalg.inv(model1['cov'])
model2['inv']=np.linalg.inv(model2['cov'])

D1 = np.array([ gaussian_pdf_multi(model1['mean'], model1['inv'], all_cl1[idx]) for idx in range(all_cl1.shape[0]) ])
D2 = np.array([ gaussian_pdf_multi(model1['mean'], model1['inv'], all_cl2[idx]) for idx in range(all_cl2.shape[0]) ])
D3 = np.array([ gaussian_pdf_multi(model1['mean'], model1['inv'], all_cl3[idx]) for idx in range(all_cl3.shape[0]) ])
D4 = np.array([ gaussian_pdf_multi(model1['mean'], model1['inv'], all_cl4[idx]) for idx in range(all_cl4.shape[0]) ])
D1_2 = np.array([ gaussian_pdf_multi(model2['mean'], model2['inv'], all_cl1[idx]) for idx in range(all_cl1.shape[0]) ])
D2_2 = np.array([ gaussian_pdf_multi(model2['mean'], model2['inv'], all_cl2[idx]) for idx in range(all_cl2.shape[0]) ])
D3_2 = np.array([ gaussian_pdf_multi(model2['mean'], model2['inv'], all_cl3[idx]) for idx in range(all_cl3.shape[0]) ])
D4_2 = np.array([ gaussian_pdf_multi(model2['mean'], model2['inv'], all_cl4[idx]) for idx in range(all_cl4.shape[0]) ])

print("D1 %0.4f"%(log_odds_summation(D1+1e-6)))
print("D2 %0.4f"%(log_odds_summation(D2+1e-6)))
print("D3 %0.4f"%(log_odds_summation(D3+1e-6)))
print("D4 %0.4f"%(log_odds_summation(D4+1e-6)))

print("D1_2 %0.4f"%(log_odds_summation(D1_2+1e-6)))
print("D2_2 %0.4f"%(log_odds_summation(D2_2+1e-6)))
print("D3_2 %0.4f"%(log_odds_summation(D3_2+1e-6)))
print("D4_2 %0.4f"%(log_odds_summation(D4_2+1e-6)))

pdb.set_trace()


# get_grid_stats(AG1[0])
pdb.set_trace()
