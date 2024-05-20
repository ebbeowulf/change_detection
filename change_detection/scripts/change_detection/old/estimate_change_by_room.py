import argparse
from accumulate_evidence import map_run
import os
import pdb
import numpy as np
import scipy
import pickle

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
            means=self.recalculate_means(num_means, current_assignments)
            new_assignments=self.get_assignments(means)
            if (new_assignments-current_assignments).sum()==0:
                break
            current_assignments=new_assignments
        err=self.weighted_error(means, current_assignments)
        return means, current_assignments, err
   
def load_grid(numpy_dir, depth_dir, num_files=100):
    MR=map_run(prompts, numpy_dir, depth_dir, initialize_model=False)
    pkl_files=[]
    for file in os.listdir(numpy_dir):
        if file.endswith(".pkl"):
             pkl_files.append(file)

    np.random.shuffle(pkl_files)
    count = 0
    for file in pkl_files:
        try:
            color_fName, global_poseM, clipV=MR.load_inference(numpy_dir+"/" + file)
            print(color_fName + " - 1" )
            MR.add_image(color_fName, global_poseM, clipV, add_ray=False)
            print(color_fName + " - 2" )
            count += 1
        except Exception as E:
            print("Error adding file - skipping")
        if count>=num_files:
            break
    
    return MR.egrid.grid    

def save_many_grids(numpy_dir, depth_dir, num_grids):
    all_grids=[]
    for idx in range(num_grids):
        G1=load_grid(numpy_dir, depth_dir)
        all_grids.append(G1)
        with open(numpy_dir+"/many_grids-intermediate.pkl","wb") as fout:
            pickle.dump(all_grids, fout)
    with open(numpy_dir+"/many_grids.pkl","wb") as fout:
        pickle.dump(all_grids, fout)
    return all_grids

prompts=["a computer", 
         "a suspicious object", 
         "signs of a break-in", 
         "a package", 
         "wooden office furniture"]

# numpy_dir1="/data2/datasets/office/no_person/initial/rotated/"
# color_dir1="/data2/datasets/office/no_person/initial/nerf_no_person_initial/images"
# depth_dir1="/data2/datasets/office/no_person/initial/depth"
# numpy_dir2="/data2/datasets/office/no_person/monitor/rotated/"
# color_dir2="/data2/datasets/office/no_person/monitor/images_combined/"
# depth_dir2="/data2/datasets/office/no_person/monitor/depth/"
# G2=load_grid(numpy_dir2, depth_dir2)
# pdb.set_trace()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('color_dir',type=str,help='location of change pose csv file to process')
    # parser.add_argument('depth_dir',type=str,help='location of change pose csv file to process')
    # parser.add_argument('numpy_results_dir',type=str,help='where the pkl files are stored')
    # parser.add_argument('--num_images',type=int,default=100, help='number of images to process (default=100)')
    # args = parser.parse_args()
    numpy_dir1="/data2/datasets/office/no_person/unchanged/rotated/"
    color_dir1="/data2/datasets/office/no_person/unchanged/images_combined"
    depth_dir1="/data2/datasets/office/no_person/unchanged/depth"
    numpy_dir2="/data2/datasets/office/no_person/bookshelf/rotated/"
    color_dir2="/data2/datasets/office/no_person/bookshelf/images_combined/"
    depth_dir2="/data2/datasets/office/no_person/bookshelf/depth/"

    save_many_grids(numpy_dir1, depth_dir1, 100)
    # pdb.set_trace()
    save_many_grids(numpy_dir2, depth_dir2, 100)
