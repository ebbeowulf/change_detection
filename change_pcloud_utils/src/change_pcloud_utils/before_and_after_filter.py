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
# from change_pcloud_utils.map_utils import identify_related_images_global_pose
from PIL import Image
import torch
from filter_clusters import cluster_filter
import cv2

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

## Build the rgbd_file_list structure for use with other pcloud utilities
def build_file_list(color_dir, depth_dir, save_dir, colmap_dir, keyword:str):
    from colmap_utils import get_all_poses
    all_poses=get_all_poses(colmap_dir,keyword=keyword)
    from colmap_utils import rgbd_file_list

    fList = rgbd_file_list(color_dir, depth_dir, save_dir, False)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # now re-order and generate files
    for key in all_poses.keys():
        uid=key.split('_')[-1].split('.')[0]
        number=int(uid)     
        fList.add_file(number,key,f"depth_{uid}.png")    
        rot=np.identity(4)
        T=np.array([[0,-1,0],[0,0,-1],[1,0,0]],dtype=float)
        rot[:3,:3]=all_poses[key]['rot_cam2world']@T #convert rotation matrix to world coordinate frame
        rot[:3,3]=all_poses[key]['pose']                
        fList.add_pose(number,rot)

    X=np.zeros((len(fList.keys())))
    Y=np.zeros((len(fList.keys())))
    for key_id, key in enumerate(fList.keys()):
        X[key_id]=fList.get_pose(key)[0,3]
        Y[key_id]=fList.get_pose(key)[1,3]

    return fList

#################
# LLM - Before And After Filter
#   Tracks the percentage of images with points inside the bounding box
#   to those with no detections. 
#################
class before_and_after_filter(cluster_filter):
    def __init__(self, use_render=True, draw_orig_box=True, flip_new=False):
        self.use_render=use_render
        self.draw_orig_box=draw_orig_box
        self.flip_new=flip_new
        if self.use_render:
            self.ba_cluster_filt=before_and_after_cluster_filter(image_scale=0.5)
            self.before_key='render'
        elif self.draw_orig_box:
            desc=['The attached image includes two side by side images of the same location captured at different times.',
                  'A box has been drawn in approximately the same location in both images to indicate the specific area of interest.'
                  'Please tell me if there is a new object contained inside the blue box on the right that was not present in the red box on the left.'
                  'If there is a new object, answer True. If the area has not changed, answer False.']

            self.ba_cluster_filt=before_and_after_cluster_filter(desc,image_scale=0.5)
            self.before_key='before'
        else:
            desc=['The attached image includes two side by side images of the same location captured at different times.',
                  'A box has been drawn in the right most image to specify the area of interest.'
                  'Please tell me if there is a new object contained inside the blue box on the right that was not present in the image on the left.'
                  'If there is a new object, answer True. If the area has not changed, answer False.']

            self.ba_cluster_filt=before_and_after_cluster_filter(desc,image_scale=0.5)
            self.before_key='before'
        self.model_name=self.ba_cluster_filt.active_model_name()
        super().__init__()

    def globalXYZ_to_imageRC(self, cam_info, tX, tY, tZ, globalM:np.array):
        invR=np.linalg.inv(globalM)
        Vi=np.matmul(invR,[tX, tY, tZ, 1])
        # Vector is in robot local space - but need to use camera coordinate frame
        #   where depth=Vi[0], col are related to Vi[1], and rows are inversely related to Vi[2]
        if Vi[0]<0:
            return -1, -1
        col=cam_info.cx-Vi[1]*cam_info.fx/Vi[0]
        row=cam_info.cy-Vi[2]*cam_info.fy/Vi[0]
        return row,col
    
    def identify_related_images_global_pose(self, 
                                            cam_info, 
                                            fList, 
                                            object_pose:np.array):
        # Need to load depth image
        valid_imgs=[]
        for key in fList.keys():      
            M=np.matmul(cam_info.rot_matrix,fList.get_pose(key))
            row,col=self.globalXYZ_to_imageRC(cam_info, object_pose[0],object_pose[1],object_pose[2],M)
            if row>=0 and row<cam_info.height and col>=0 and col<cam_info.width:
                valid_imgs.append(key)
        return valid_imgs
  
    def set_closest_images(self, all_images, cluster):
        global fList_base, fList_new, params

        # find images in the baseline dataset that can see the target cluster
        rel_imgs=self.identify_related_images_global_pose(params, fList_base, cluster.centroid)
        # now find the vector to the object and the associated distance
        dist_relM=np.zeros((len(rel_imgs)))
        zVec_relM=np.zeros((3,len(rel_imgs)))
        for key_idx, rel_key in enumerate(rel_imgs):
            relM=np.matmul(params.rot_matrix,fList_base.get_pose(int(rel_key)))
            dist_relM[key_idx]=np.sqrt(((cluster.centroid[:3]-relM[:3,3])**2).sum())
            zVec_relM[:,key_idx]=np.matmul(relM[:3,:3],[1,0,0])
        zVec_relM=np.transpose(zVec_relM)

        # step through images that point at the object and calculate     
        for key in all_images.keys():
            if 'new' in all_images[key]:
                M=np.matmul(params.rot_matrix,fList_new.get_pose(int(key)))
                distM=np.sqrt(((cluster.centroid-M[:3,3])**2).sum())
                zVecM=np.matmul(M[:3,:3],[1,0,0])
                angle=np.arccos(np.dot(zVec_relM,zVecM)) #dot-product, returns a [N,] vector
                best_match=rel_imgs[np.argmax(gaussian_dist_function(dist_relM-distM,1.0)*gaussian_dist_function(angle,0.25))]
                color_fName=fList_base.get_color_fileName(int(best_match))
                print(f"Best Match{key}: {color_fName}")
                try:
                    img=Image.open(color_fName)
                except Exception as e:
                    print(f"Cannot load image {color_fName}- skipping")

                np_img=np.array(img)
                if self.draw_orig_box:                
                    # Need to draw a box around the object
                    #   going to calculate the R/C for every corner of the 3D box
                    coord=np.zeros((8,2),dtype=float)
                    baseM=fList_base.get_pose(best_match)
                    coord[0,0],coord[0,1]=self.globalXYZ_to_imageRC(params,cluster.box[0,0],cluster.box[0,1],cluster.box[0,2],baseM)
                    coord[1,0],coord[1,1]=self.globalXYZ_to_imageRC(params,cluster.box[1,0],cluster.box[0,1],cluster.box[0,2],baseM)
                    coord[2,0],coord[2,1]=self.globalXYZ_to_imageRC(params,cluster.box[0,0],cluster.box[1,1],cluster.box[0,2],baseM)
                    coord[3,0],coord[3,1]=self.globalXYZ_to_imageRC(params,cluster.box[0,0],cluster.box[0,1],cluster.box[1,2],baseM)
                    coord[4,0],coord[4,1]=self.globalXYZ_to_imageRC(params,cluster.box[1,0],cluster.box[1,1],cluster.box[0,2],baseM)
                    coord[5,0],coord[5,1]=self.globalXYZ_to_imageRC(params,cluster.box[0,0],cluster.box[1,1],cluster.box[1,2],baseM)
                    coord[6,0],coord[6,1]=self.globalXYZ_to_imageRC(params,cluster.box[1,0],cluster.box[0,1],cluster.box[1,2],baseM)
                    coord[7,0],coord[7,1]=self.globalXYZ_to_imageRC(params,cluster.box[1,0],cluster.box[1,1],cluster.box[1,2],baseM)
                    coord=coord.astype('int')
                    def clip_boundary(val,maxV):
                        return min(maxV,max(0,val))
                        
                    new_box=[[clip_boundary(coord[:,1].min()-10,np_img.shape[1]),clip_boundary(coord[:,0].min()-10,np_img.shape[0])],
                            [clip_boundary(coord[:,1].max()+10,np_img.shape[1]),clip_boundary(coord[:,0].max()+10,np_img.shape[0])]]
                    np_img=cv2.rectangle(np_img, new_box[0], new_box[1], (255,0,0), 5)
                all_images[key]['before']=np_img
        return all_images
    
    def calculate_score(self,all_images, cluster=None):
        if not self.use_render: # need to add "before" images to all_images
            all_images=self.set_closest_images(all_images,cluster)
        # Need to flip dimensions of the images - they are inverted
        for key in all_images.keys():
            if 'new' in all_images[key]:
                all_images[key]['new']=all_images[key]['new'][:,:,[2,1,0]]
                if self.flip_new:
                    all_images[key]['new']=cv2.rotate(all_images[key]['new'],cv2.ROTATE_180)
            if 'render' in all_images[key]:
                all_images[key]['render']=all_images[key]['render'][:,:,[2,1,0]]
        return self.ba_cluster_filt.evaluate_multiple_image_pairs(all_images, before_key=self.before_key)

    def get_save_file_name(self, tgt_dir, prompt):
        if self.use_render:
            return f"{tgt_dir}/render_and_after_filter-{prompt}-{self.model_name}.json"
        if not self.draw_orig_box:
            return f"{tgt_dir}/before_unannotated_and_after_filter-{prompt}-{self.model_name}.json"
        return f"{tgt_dir}/before_and_after_filter-{prompt}-{self.model_name}-{self.model_name}.json"
    
    def clear_results(self):
        self.all_results=dict()
        return super().clear_results()
    
    def get_score(self, cluster_id, all_images=None, cluster=None):
        if cluster_id not in self.scores:
            if all_images is None:
                raise Exception("Cannot score cluster without a valid all_images struct")
            self.scores[cluster_id]=self.calculate_score(all_images, cluster)
            self.all_results[cluster_id]=self.ba_cluster_filt.all_results
        return self.scores[cluster_id]
    
    def serialize(self):
        filter_state=super().serialize()
        filter_state['all_results']=self.all_results
        return filter_state
    
    def loader(self, json_dict):
        super().loader(json_dict)
        self.all_results=json_dict['all_results']

def score_all_clusters(tgt_dir, query, filterName, flip_new):
    global fList_new
    Q=query.replace(" ","_")
    check_str=os.path.join(tgt_dir,Q+"*[0-9].pkl")
    cluster_files=glob.glob(check_str)    

    # Initialize the filterBank - loading from file when possible
    filterBank={}
    if filterName=='before_and_after_filter':
        filterBank[filterName]=before_and_after_filter(use_render=False, flip_new=flip_new)
    elif filterName=='render_and_after_filter':
        filterBank[filterName]=before_and_after_filter(use_render=True) # flip_new is not needed when rendering images
    elif filterName=='before_unannotated_and_after_filter':
        filterBank[filterName]=before_and_after_filter(use_render=False,draw_orig_box=False, flip_new=flip_new)
    else:
        raise Exception(f"Unknown filter type: {filterName}")
    
    filterBank[filterName].loadFromFile(tgt_dir, Q)

    # Score all of the clusters - storing the results locally in the filter
    #   by the ID of the cluster
    print(f"Evaluating {query}")
    for file in cluster_files:
        # file='/data2/datasets/s120/T2/changes/change1/save_results_unfiltered/sam3_0.1_openVocab/small_items_7.pkl'
        with open(file, 'rb') as handle:
            A=pickle.load(handle)     
        ID=file.split('_')[-1].split('.')[0]
        filterBank[filterName].get_score(ID,A['all_images'],A['cluster'])

    # Save the result
    filterBank[filterName].save2file(tgt_dir, Q)

    return filterBank

def setup_before_and_after_filter(initial_dir, 
                                  nerfacto_dir,
                                  color_dir="images_combined",
                                  colmap_dir="nerf_colmap/colmap/sparse_geo/0",
                                  frame_keyword="color"):
    # Need to build information from the baseline run 
    #   Specifically we need the fList_base and params variables to be created globally
    # global fList_base, params
    from colmap_utils import get_camera_params
    
    # initial_dir=nerfacto_dir.split('outputs')[0]
    # initial_colmap_dir=initial_dir+colmap_dir
    global fList_base, params, fList_new
    params=get_camera_params(colmap_dir,nerfacto_dir)
    fList_base=build_file_list(color_dir,"",initial_dir,colmap_dir,frame_keyword)
    fList_new=build_file_list(color_dir,"",initial_dir,colmap_dir,"new")
    if len(fList_base.keys())==0:
        print("No images found in the base directory - check your frame keyword?")
        raise(Exception("fList_base empty"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('tgt_dir', type=str, help='Directory to evaluate - should already contain cluster pkl files')
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dataparser_transforms.json.')
    parser.add_argument('--queries', type=str, nargs='*', help="List of queries to evaluate")
    ## These parameters are all for the before and after filter
    parser.add_argument('--color_dir',type=str,default='../../images_combined',help='where are the color images of the base directory? (default=../../colmap_combined)')
    parser.add_argument('--colmap_dir',type=str,default='../../colmap_combined/sparse_combined/0', help='where are the images + cameras.txt files? (default = ../../colmap_combined/sparse_combined/0/)')
    parser.add_argument('--frame_keyword',type=str,default="color",help='a keyword to use when parsing the transforms file to find base directory images (default=color)')    
    parser.add_argument('--filterType', type=str, default='before_and_after_filter', help='Specify the filter type (supports before_and_after_filter (default), before_unannotated_and_after_filter, render_and_after_filter )')
    parser.add_argument('--rotate_180', dest='flip_new', action='store_true', help='Rotate the input image 180 degrees to match the baseline')
    parser.set_defaults(flip_new=False)
    args = parser.parse_args()

    setup_before_and_after_filter(args.tgt_dir, args.nerfacto_dir, args.tgt_dir + "/" + args.color_dir, args.tgt_dir+"/"+args.colmap_dir, args.frame_keyword)
    allQ=args.queries
    for query in args.queries:
        fBank=score_all_clusters(args.tgt_dir, query, args.filterType, args.flip_new)
        ids = list(fBank[args.filterType].scores.keys())
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


