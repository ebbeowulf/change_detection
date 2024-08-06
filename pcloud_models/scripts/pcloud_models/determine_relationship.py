from object_pcloud_from_scannet import read_label_csv, id_from_string, load_camera_info, retrieve_object_pcloud, build_file_structure, get_scene_type
import open3d as o3d
import pdb
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from farthest_point_sampling.fps import farthest_point_sampling
from rgbd_file_list import rgbd_file_list
from camera_params import camera_params
import pickle

# FURNITURE=['chair','couch','potted plant','bed','mirror','dining table','window','desk','toilet','door']
FURNITURE=['cabinet','chair','couch','plant','bed','dining table','desk','toilet','ottoman'] # door + window have too many errors
APPLIANCE=['tv','microwave','oven','toaster','refrigerator','sink'] #'blender' is in COCO, but not scannet

DBSCAN_MIN_SAMPLES=20 
DBSCAN_GRIDCELL_SIZE=0.01
DBSCAN_EPS=DBSCAN_GRIDCELL_SIZE*2.5
CLUSTER_MIN_COUNT=10000
CLUSTER_PROXIMITY_THRESH=0.3
CLUSTER_TOUCHING_THRESH=0.05

def get_distinct_clusters(pcloud, gridcell_size=DBSCAN_GRIDCELL_SIZE, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, cluster_min_count=CLUSTER_MIN_COUNT, floor_threshold=0.1):
    clouds=[]
    if pcloud is None or len(pcloud.points)<CLUSTER_MIN_COUNT:
        return clouds
    pcd_small=pcloud.voxel_down_sample(gridcell_size)
    p2=DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(pcd_small.points))
    pts=np.array(pcd_small.points)

    for id in range(10): # don't bother with counts>10
        whichP=(p2.labels_==id)
        if whichP.sum()>cluster_min_count:
            pts2=pts[whichP]
            whichP2=(pts2[:,2]>floor_threshold)
            if whichP2.sum()>cluster_min_count:
                clouds.append(object_pcloud(pts2[whichP2]))

    return clouds

class object_pcloud():
    def __init__(self, pts, label:str=None, num_samples=1000):
        self.box=np.vstack((pts.min(0),pts.max(0)))
        self.pts=pts
        self.label=label
        self.farthestP=farthest_point_sampling(self.pts, num_samples)

    def set_label(self, label):
        self.label=label
    
    def is_box_overlap(self, input_cloud, dimensions=[0,1,2], threshold=0.3):
        for dim in dimensions:
            if self.box[1,dim]<(input_cloud.box[0,dim]-threshold) or self.box[0,dim]>=(input_cloud.box[1,dim]+threshold):
                return False
        return True

    def compute_cloud_distance(self, input_cloud):
        input_pt_matrix=input_cloud.pts[input_cloud.farthestP]
        min_sq_dist=1e10
        for pid in self.farthestP:
            min_sq_dist=min(min_sq_dist, ((input_pt_matrix-self.pts[pid])**2).sum(1).min())
        return np.sqrt(min_sq_dist)
    
    def is_above(self, input_cloud):
        # check centroid location relative to bounding box
        ctr=self.pts.mean(0)

        # Should be overlapped in x + y directions
        if ctr[0]>input_cloud.box[0,0] and ctr[0]<input_cloud.box[1,0] and ctr[1]>input_cloud.box[0,1] and ctr[1]<input_cloud.box[1,1]:
            # Should also be "above" the other centroid
            input_ctr=input_cloud.pts.mean(0)
            return ctr[2]>input_ctr[2]
        return False
    
class determine_relationship():
    def __init__(self, categories:list=None):
        if categories is None:
            self.categories=FURNITURE
            self.categories=self.categories+APPLIANCE
        else:
            self.categories=categories
        
        read_label_csv()
        self.category_id=[]
        self.pclouds=dict()
        for lbl_ in self.categories:
            self.category_id = self.category_id + id_from_string(lbl_)

    def add_pcloud(self, cloud_dict, object_type:str):
        uid=np.random.randint(1e10)
        while uid in self.pclouds:
            uid=np.random.randint(1e10)
        
        self.pclouds[uid]=cloud_dict
        self.pclouds[uid].set_label(object_type)

    def load_furniture(self, fList:rgbd_file_list, params_fileName):
        params=load_camera_info(params_fileName)
        self.sceneType=get_scene_type(params_fileName)

        self.pclouds=dict()
        for lbl_ in self.categories:
            pcd=retrieve_object_pcloud(params, fList, lbl_)
            clusters=get_distinct_clusters(pcd)
            for cls_ in clusters:
                self.add_pcloud(cls_, lbl_)
        
        key_list=list(self.pclouds.keys())
        self.furniture_relationships=[]
        for idxA, uidA in enumerate(key_list):
            for uidB in key_list[(idxA+1):]:
                print("Checking relationship %s(%d) <-> %s(%d)"%(self.pclouds[uidA].label, uidA, self.pclouds[uidB].label, uidB))

                # Is uidA above uidB?
                distance=self.pclouds[uidA].compute_cloud_distance(self.pclouds[uidB])
                if self.pclouds[uidA].is_above(self.pclouds[uidB]):
                    if distance<CLUSTER_TOUCHING_THRESH:
                        self.furniture_relationships.append([uidA,uidB,"on"])
                    else:
                        self.furniture_relationships.append([uidA,uidB,"above"])
                elif self.pclouds[uidB].is_above(self.pclouds[uidA]): #is uidB above uidA?
                    if distance<CLUSTER_TOUCHING_THRESH:
                        self.furniture_relationships.append([uidB,uidA,"on"])
                    else:
                        self.furniture_relationships.append([uidB,uidA,"above"])                
                elif distance<CLUSTER_PROXIMITY_THRESH:  # are they close enough in the horizontal to be 'next to'
                    self.furniture_relationships.append([uidA, uidB, "next to"])
                else:
                    print("Not related")
        print(self.furniture_relationships)

    def load_objects(self, fList:rgbd_file_list, tgt_class:str):
        pcloud_fName=fList.get_combined_raw_fileName(tgt_class)

        try:
            with open(pcloud_fName, 'rb') as handle:
                self.object_pcloud=pickle.load(handle)
        except Exception as e:
            print(f"Loading pcloud from {pcloud_fName} failed ... ")
            self.object_pcloud = None
            return
        
    # Reprocess the object pointcloud to count clusters and generate
    #   relationships between candidate objects and known furniture
    def process_objects(self, tgt_class:str, initial_threshold:float, min_cluster_size=1000, floor_threshold=0.05):
        if self.object_pcloud is None or 'xyz' not in self.object_pcloud or 'probs' not in self.object_pcloud:
            print(f"Error processing object pointcloud for {tgt_class}")
        
        pdb.set_trace()
        whichP=(self.object_pcloud['probs']>=initial_threshold)
        pcd=o3d.geometry.PointCloud()
        xyzF=self.object_pcloud['xyz'][whichP]
        F2=np.where(np.isnan(xyzF).sum(1)==0)        
        pcd.points=o3d.utility.Vector3dVector(xyzF[F2])
        object_clusters=get_distinct_clusters(pcd, cluster_min_count=min_cluster_size, floor_threshold=floor_threshold)
        pdb.set_trace()

    def get_room_statement(self):
        if self.sceneType is not None:
            return f"I am in a {self.sceneType}."
        return ""

    def get_furniture_statement(self):
        # count objects
        o_count={}
        for id in self.pclouds:
            if self.pclouds[id].label not in o_count:
                o_count[self.pclouds[id].label] = 0
            o_count[self.pclouds[id].label]+=1
        
        # generate statement to summarize objects
        object_stmt="There is "
        all_object_types=list(o_count.keys())
        for lbl in all_object_types:
            is_last=(lbl==all_object_types[-1])
            # Add the first object with 'a' or 'and' for the last in the list
            if is_last and o_count[lbl]==1:
                object_stmt+=f"and a {lbl}."
            else:
                object_stmt+=f"a {lbl}, "

            # Following objects are added with 'another' to avoid using plurals
            for lbl_cnt in range(1,o_count[lbl]):
                if is_last and lbl_cnt==(o_count[lbl]-1):
                    object_stmt+=f" and another {lbl}."
                else:
                    object_stmt+=f"another {lbl}, "

        return object_stmt
            
    def get_furniture_relationships(self):
        # Now describe relationships
        rel_stmts=[]
        for rel in self.furniture_relationships:
            stmt=f"There is a {self.pclouds[rel[0]].label} {rel[2]} the {self.pclouds[rel[1]].label}."
            if stmt in rel_stmts:
                stmt=f"There is another {self.pclouds[rel[0]].label} {rel[2]} the {self.pclouds[rel[1]].label}."
            rel_stmts.append(stmt)
        
        for stmt in rel_stmts:
            combined_stmt+=f" {stmt}"
        
        return combined_stmt

    # def build_object_statements(self, tgt_class, threshold_list):
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('tgt_class',type=str,help='class to build descriptive file for')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--label_dir',type=str,default='label-filt', help='subdirectory containing the label images')
    args = parser.parse_args()
   
    save_dir=args.root_dir+"/"+args.save_dir
    fList=build_file_structure(args.root_dir+"/"+args.raw_dir, args.root_dir+"/"+args.label_dir, save_dir)

    s_root=args.root_dir.split('/')
    if s_root[-1]=='':
        par_file=args.root_dir+"%s.txt"%(s_root[-2])
    else:
        par_file=args.root_dir+"/%s.txt"%(s_root[-1])

    rel=determine_relationship()
    rel.load_furniture(fList, par_file)
    rel.load_objects(fList, args.tgt_class)
    rel.process_objects(args.tgt_class, 0.5)
    combined_stmt=rel.get_furniture_description()
    print(combined_stmt)
