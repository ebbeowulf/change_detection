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
import sys

# FURNITURE=['chair','couch','potted plant','bed','mirror','dining table','window','desk','toilet','door']
FURNITURE=['cabinet','chair','couch','plant','bed','dining table','desk','toilet','ottoman'] # door + window have too many errors
APPLIANCE=['tv','microwave','oven','toaster','refrigerator','sink'] #'blender' is in COCO, but not scannet


class determine_relationship():
    def __init__(self, categories:list=None):
        if categories is None:
            self.categories=FURNITURE
            self.categories=self.categories+APPLIANCE
        else:
            self.categories=categories
        
        read_label_csv()
        self.category_id=[]
        self.furniture_pclouds=dict()
        self.object_raw=None
        self.object_pclouds=dict()
        for lbl_ in self.categories:
            self.category_id = self.category_id + id_from_string(lbl_)

    # Add another object to the long-term object map
    #   These are usually furniture or other objects that we 
    #   have good recognition models for (i.e. Yolo or hand labeled)
    def add_furniture(self, cloud_dict, object_type:str):
        uid=np.random.randint(1e10)
        while uid in self.furniture_pclouds:
            uid=np.random.randint(1e10)
        
        self.furniture_pclouds[uid]=cloud_dict
        self.furniture_pclouds[uid].set_label(object_type)

    # Load the set of known furniture for the environment
    #   and then determine the relationships between these furniture
    #   The only relationships used are on, over and next to.
    def load_furniture(self, fList:rgbd_file_list, params_fileName):
        params=load_camera_info(params_fileName)
        self.sceneType=get_scene_type(params_fileName)

        self.furniture_pclouds=dict()
        for lbl_ in self.categories:
            pcd=retrieve_object_pcloud(params, fList, lbl_)
            clusters=get_distinct_clusters(pcd)
            print(f"Counted {len(clusters)} {lbl_}")
            for cls_ in clusters:
                self.add_furniture(cls_, lbl_)
        
    def process_furniture(self):
        key_list=list(self.furniture_pclouds.keys())
        self.furniture_relationships=[]
        for idxA, uidA in enumerate(key_list):
            for uidB in key_list[(idxA+1):]:
                print("Checking relationship %s(%d) <-> %s(%d)"%(self.furniture_pclouds[uidA].label, uidA, self.furniture_pclouds[uidB].label, uidB))

                # Is uidA above uidB?
                distance=self.furniture_pclouds[uidA].compute_cloud_distance(self.furniture_pclouds[uidB])
                if self.furniture_pclouds[uidA].is_above(self.furniture_pclouds[uidB]):
                    if distance<CLUSTER_TOUCHING_THRESH:
                        self.furniture_relationships.append([uidA,uidB,"on"])
                    else:
                        self.furniture_relationships.append([uidA,uidB,"above"])
                elif self.furniture_pclouds[uidB].is_above(self.furniture_pclouds[uidA]): #is uidB above uidA?
                    if distance<CLUSTER_TOUCHING_THRESH:
                        self.furniture_relationships.append([uidB,uidA,"on"])
                    else:
                        self.furniture_relationships.append([uidB,uidA,"above"])                
                elif distance<CLUSTER_PROXIMITY_THRESH:  # are they close enough in the horizontal to be 'next to'
                    self.furniture_relationships.append([uidA, uidB, "next to"])
                else:
                    print("Not related")
        print(self.furniture_relationships)

    # Load the raw point cloud used to determine object hypotheses
    def load_objects(self, fList:rgbd_file_list, tgt_class:str):
        pcloud_fName=fList.get_combined_raw_fileName(tgt_class)

        try:
            with open(pcloud_fName, 'rb') as handle:
                self.object_raw=pickle.load(handle)
        except Exception as e:
            print(f"Loading pcloud from {pcloud_fName} failed ... ")
            self.object_raw = None
            return
        
    # Reprocess the object pointcloud to count clusters and generate
    #   relationships between candidate objects and known furniture
    def process_objects(self, tgt_class:str, initial_threshold:float, min_cluster_size=1000, floor_threshold=0.05):
        if self.object_raw is None or 'xyz' not in self.object_raw or 'probs' not in self.object_raw:
            print(f"Error processing object pointcloud for {tgt_class}")
        
        whichP=(self.object_raw['probs']>=initial_threshold)
        pcd=o3d.geometry.PointCloud()
        xyzF=self.object_raw['xyz'][whichP]
        F2=np.where(np.isnan(xyzF).sum(1)==0)
        xyzF2=xyzF[F2]        
        pcd.points=o3d.utility.Vector3dVector(xyzF2)
        object_clusters=get_distinct_clusters(pcd, cluster_min_count=min_cluster_size, floor_threshold=floor_threshold)
        self.object_pclouds=dict()
        if len(object_clusters)>0:
            # Create the object pclouds object - stores hypothetical locations for the target object
            #   in a list of the same style as furniture pclouds
            probF=self.object_raw['probs'][whichP]
            probF2=probF[F2]
            for idx, cl_ in enumerate(object_clusters):
                idxO=f"L{idx}"
                self.object_pclouds[idxO]=cl_
                # Estimate the probability of each cluster using max + mean
                self.object_pclouds[idxO].estimate_probability(xyzF2,probF2)
            
            if 0: # draw the result
                rgbF=self.object_raw['rgb'][whichP]
                pcd.colors = o3d.utility.Vector3dVector(rgbF[F2][:,[2,1,0]]/255) 
                self.draw_pcloud(pcd)

            # Now in a similar process to that used to describe furniture, identify relationships between
            #   possible object locations and furniture
            furniture_key_list=list(self.furniture_pclouds.keys())
            object_key_list=list(self.object_pclouds.keys())
            self.object_relationships=[]
            for idxO, uidO in enumerate(object_key_list):
                for idxF, uidF in enumerate(furniture_key_list):
                    print("Checking relationship %s(%s) <-> %s(%d)"%(self.object_pclouds[uidO].label, uidO, self.furniture_pclouds[uidF].label, uidF))

                    # Is uidA above uidB?
                    distance=self.object_pclouds[uidO].compute_cloud_distance(self.furniture_pclouds[uidF])
                    if self.object_pclouds[uidO].is_above(self.furniture_pclouds[uidF]):
                        if distance<CLUSTER_TOUCHING_THRESH:
                            self.object_relationships.append([uidO,uidF,"on"])
                        else:
                            self.furniture_relationships.append([uidO,uidF,"above"])
                    elif self.furniture_pclouds[uidF].is_above(self.object_pclouds[uidO]): #is uidB above uidA?
                        self.furniture_relationships.append([uidO,uidF,"under"])                
                    elif distance<CLUSTER_PROXIMITY_THRESH:  # are they close enough in the horizontal to be 'next to'
                        self.furniture_relationships.append([uidO, uidF, "next to"])
                    else:
                        print("Not related")
            print(self.object_relationships)

    def get_room_statement(self):
        if self.sceneType is not None:
            return f"I am in a {self.sceneType}."
        return ""

    def get_furniture_statement(self):
        # count objects
        o_count={}
        for id in self.furniture_pclouds:
            if self.furniture_pclouds[id].label not in o_count:
                o_count[self.furniture_pclouds[id].label] = 0
            o_count[self.furniture_pclouds[id].label]+=1
        
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
            stmt=f"There is a {self.furniture_pclouds[rel[0]].label} {rel[2]} the {self.furniture_pclouds[rel[1]].label}."
            if stmt in rel_stmts:
                stmt=f"There is another {self.furniture_pclouds[rel[0]].label} {rel[2]} the {self.furniture_pclouds[rel[1]].label}."
            rel_stmts.append(stmt)
        
        combined_stmt=""
        for stmt in rel_stmts:
            combined_stmt+=f" {stmt}"
        
        return combined_stmt

    def build_object_statements(self, tgt_class):
        if len(self.object_pclouds)<1:
            return ""
        
        object_key_list=list(self.object_pclouds.keys())
        combined_stmt=""
        for uidO in object_key_list:
            combined_stmt+=f"There is a possible {tgt_class}"
            # Build the list of relationships
            rset=[]
            for rel in self.object_relationships:
                if rel[0]==uidO:
                    rset.append(rel)
                        
            if self.object_pclouds[uidO].box[0,2]<CLUSTER_PROXIMITY_THRESH:
                combined_stmt+=" on the floor"

            if len(rset)==0:
                combined_stmt+=" next to none of the objects"
            else:
                if len(rset)>1:
                    for rel in rset[:-1]:
                        combined_stmt+=f" {rel[2]} the {self.furniture_relationships[rel[1]]} and"

                combined_stmt+=f" {rset[-1][2]} the {self.furniture_pclouds[rset[-1][1]].label}"
            combined_stmt+=f" at location {uidO}. "
        return combined_stmt

    def create_map_summary(self, tgt_class:str, confidence_values:list):
        summary={}
        summary['room']=self.get_room_statement()
        summary['furniture']=self.get_furniture_statement()
        summary['furniture_relationships']=self.get_furniture_relationships()
        summary['object_results']={}
        for conf_threshold in confidence_values:
            dkey=f"{conf_threshold:.2f}"
            summary['object_results'][dkey]=dict()
            self.process_objects(tgt_class, conf_threshold)
            summary['object_results'][dkey]['object_list']=list(self.object_pclouds.keys())
            summary['object_results'][dkey]['max_prob']  = [ self.object_pclouds[uidO].prob_stats['max'] for uidO in self.object_pclouds.keys() ]
            summary['object_results'][dkey]['mean_prob'] = [ self.object_pclouds[uidO].prob_stats['mean'] for uidO in self.object_pclouds.keys() ]
            summary['object_results'][dkey]['stdev'] = [ self.object_pclouds[uidO].prob_stats['stdev'] for uidO in self.object_pclouds.keys() ]
            summary['object_results'][dkey]['pcount'] = [ self.object_pclouds[uidO].prob_stats['pcount'] for uidO in self.object_pclouds.keys() ]
            summary['object_results'][dkey]['boxes'] = [ self.object_pclouds[uidO].box.tolist() for uidO in self.object_pclouds.keys() ]
            summary['object_results'][dkey]['combined_statement']=self.build_object_statements(tgt_class)
        return summary

    def draw_pcloud(self, pcloud, draw_furniture=True, draw_objects=True, draw_labels=False):
        obj_boxes=[ self.object_pclouds[uidO].box.tolist() for uidO in self.object_pclouds.keys() ]
        f_boxes=[ self.furniture_pclouds[uidO].box.tolist() for uidO in self.furniture_pclouds.keys() ]
        labels=[ self.furniture_pclouds[uidO].label for uidO in self.furniture_pclouds.keys() ]
        
        from draw_pcloud import drawn_image
        image_interface=drawn_image(pcloud)        
        if draw_furniture:
            image_interface.add_boxes_to_fg(f_boxes)
        if draw_objects:
            image_interface.add_boxes_to_fg(obj_boxes,(0,255,0))
        image_interface.draw_fg()

    def display_map(self,fList:rgbd_file_list, draw_furniture=True, draw_objects=True, draw_labels=False):
        ply_fileName=fList.get_combined_pcloud_fileName()
        try:
            pcl_target=o3d.io.read_point_cloud(ply_fileName)
            if pcl_target is None:
                print("Pcloud not found - failed to draw")
                return
        except Exception as e:
            print("Pcloud not found - failed to draw")
            return None

        self.draw_pcloud(pcl_target, draw_furniture, draw_objects, draw_labels)    
        pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('tgt_class',type=str,help='class to build descriptive file for')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--label_dir',type=str,default='label-filt', help='subdirectory containing the label images')
    parser.add_argument('--save_json',type=str,default=None, help='full path to save json summary file (default={save_dir}/{tgt_class}.summary.json)')
    parser.add_argument('--draw', dest='draw', action='store_true')
    parser.set_defaults(draw=False)
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
    if args.draw:
        # Cannot both draw and do the map summary thing ... go ahead and exit
        rel.process_objects(args.tgt_class, 0.01)
        rel.display_map(fList)
        sys.exit(-1)

    rel.process_furniture()    
    map_sum=rel.create_map_summary(args.tgt_class,np.arange(0.5,0.99,0.02))

    if args.save_json is None:
        save_json=fList.get_json_summary_fileName(args.tgt_class)
    else:
        save_json=args.save_json
    
    import json
    with open(save_json,'w') as fout:
        json.dump(map_sum,fout)