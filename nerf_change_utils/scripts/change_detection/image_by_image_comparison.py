from image_set import image_set
import os
from PIL import Image
import pdb
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from clustering import cluster, cluster_history
import pickle
from grid import max_grid3D

# ROOT_RAW_DIR="/data2/datasets/office/no_person/"
# ROOT_NERF_DIR="/home/emartinso/projects/nerfstudio/renders/no_person_"

# DBSCAN Parameters
DEPTH_CLUSTER_EPS=0.1          
DEPTH_CLUSTER_MINC=1000 #200 #1000
MAX_CLUSTERING_SAMPLES=100000 #DBSCAN struggles to handle more points than this, so randomly sample after this threshold

K_ROTATED=[906.7647705078125, 0.0, 368.2167053222656, 
                0.0, 906.78173828125, 650.24609375,                 
                0.0, 0.0, 1.0]
F_X=K_ROTATED[0]
C_X=K_ROTATED[2]
F_Y=K_ROTATED[4]
C_Y=K_ROTATED[5]

def get_3D_point(row, col, depth):
    x = (col - C_X) * depth / F_X
    y = (row - C_Y) * depth / F_Y
    return np.vstack((x,y,depth))

def calculate_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    value = cv2.Laplacian(gray, cv2.CV_64F).var()
    return value

def get_rc_from_xyz(x, y, z, cam_poseM):
    xyz=np.vstack((x,y,z))
    pt2=np.matmul(cam_poseM[:3,:3].transpose(),(xyz-np.expand_dims(cam_poseM[:3,3],1)))    
    col=((pt2[0]*F_X/pt2[2])+C_X).astype(int)
    row=((pt2[1]*F_Y/pt2[2])+C_Y).astype(int)
    return np.vstack((row,col)).transpose()

def get_center_point(depthI):
    if depthI is None:
        return None
    half_s=np.round(np.array(depthI.shape)/2.0).astype('int')
    subD=depthI[(half_s[0]-10):(half_s[0]+10),(half_s[1]-10):(half_s[1]+10)]
    rc=np.where(subD>0.1)
    center_depth=np.median(subD[rc])
    return get_3D_point(half_s[0],half_s[1],center_depth)

class image_comparator():
    def __init__(self, 
                 images_initial_csv:str, 
                 images_secondary_csv:str, 
                 detection_target: str,
                 is_positive_change:bool,
                 #  use_nerf:bool, 
                 use_clip:bool,
                 min_cluster_count=DEPTH_CLUSTER_MINC):

        # self.use_nerf=use_nerf
        self.use_clip=use_clip
        self.detection_target=detection_target
        self.is_positive_change=is_positive_change
        if use_clip:
            from clip_segmentation import clip_seg
            self.detect=clip_seg([self.detection_target])
        else:
            from yolo_segmentation import yolo_segmentation
            self.detect=yolo_segmentation()
        self.detect_id=None
        self.images1=image_set(images_initial_csv)
        self.images2=image_set(images_secondary_csv)
        self.cluster_min_count=min_cluster_count

        # If we are not using nerf, then we need to build
        #   a list of clusters from the initial dataset
        #   these should be loaded to/from file most of the time
        self.initial_cluster_history=None # creation done on first call to build_eg_change

    def change_cluster_min_count(self, min_cluster_count):
        self.cluster_min_count=min_cluster_count

    def get_object_detection_data(self, image:str):
        cv_image=self.detect.process_file(image)
        if self.detect_id is None:
            self.detect_id=self.detect.get_id(self.detection_target)
        clipV=self.detect.get_prob_array(self.detect_id)
        if clipV is None:
            clipV=np.zeros(cv_image.shape[0:2],dtype=float)
        return cv_image, clipV

    # def color_image_name(self, image_name, ):
    #     return ROOT_RAW_DIR+image_dir_and_name[0]+"/rotated/" + image_dir_and_name[1]
    
    # Load the depth image with the assumption that depth images are of the format <dir>/depth/depth_<id>.png
    def load_depth_image(self, depth_image_directory:str, image_number:str): 
        # number=image_dir_and_name[1].split('_')[1]
        # Load the depth image
        depth_image_name=depth_image_directory + "/depth_" + image_number
        depth_image = cv2.imread(depth_image_name, -1)
        depth_image=cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
        return depth_image.astype(float)/1000.0

    # For each specified xy in pixel coordinates, create a 3D point from reading the depth image
    def create_pcloud(self, image_dir_and_name:list, pts_xy, depth_image):
        # Get the untransformed points
        D=depth_image[pts_xy]
        d_mask=np.where(D>0.1) # filter out the zero depth points
        pts=get_3D_point(pts_xy[0][d_mask],pts_xy[1][d_mask],D[d_mask])
        # Transform to map space using the image calibration info
        pts=np.vstack((pts,np.ones((1,pts.shape[1]),dtype=float)))
        # Check both the initial set and the secondary set to find this image
        cam_poseM=self.images2.get_pose_by_name(image_dir_and_name[0],image_dir_and_name[1])
        if cam_poseM is None:
            cam_poseM=self.images1.get_pose_by_name(image_dir_and_name[0],image_dir_and_name[1])
        pts=np.matmul(cam_poseM,pts)[:3,:].transpose()
        return pts, d_mask

    # For each specified xy in pixel coordinates, create a 3D point from reading the depth image
    def get_egrid_boundaries(self, image_dir_and_name:list, depth_image):
        xy=np.where((depth_image>0.1)*(depth_image<5000))
        whichP=np.random.choice(np.arange(xy[0].shape[0]),10000)
        pts_xy=(xy[0][whichP],xy[1][whichP])
        pts,__=self.create_pcloud(image_dir_and_name,pts_xy,depth_image)        
        pts_mn=pts.min(0)-0.1
        pts_mx=pts.max(0)+0.1
        # count dimensions to achieve cell size of ~5 cm        
        num_dim=((pts_mx-pts_mn)/0.05).astype(int)
        return pts_mn, pts_mx, num_dim
    
    # Reduce very large arrays for use by clustering - reduction done by random sampling
    #   up to maximum
    def sample_pts(self, grid_pts, scores):
        if grid_pts.shape[0]<MAX_CLUSTERING_SAMPLES:
            return grid_pts, scores
        
        rr=np.random.choice(np.arange(grid_pts.shape[0]),size=MAX_CLUSTERING_SAMPLES)
        return grid_pts[rr], scores[rr]
    
    # cluster the specified points
    def apply_dbscan_clustering(self, grid_pts, scores, eps, min_count):
        if grid_pts is None or grid_pts.shape[0]<5:
            return []
        gp_sampled, sc_sampled=self.sample_pts(grid_pts, scores)
        CL2=DBSCAN(eps=eps, min_samples=min_count).fit(gp_sampled,sample_weight=sc_sampled)
        clusters=[]
        for idx in range(10):
            whichP=np.where(CL2.labels_== idx)            
            if len(whichP[0])<1:
                break
            clusters.append(cluster(gp_sampled[whichP],sc_sampled[whichP]))
        return clusters

    def check_cluster_blur(self, cl_:cluster, cam_poseM):
        #Recover pixel coordinates
        try:
            c2=np.vstack((cl_.min(),cl_.max()))
            corners=get_rc_from_xyz(c2[:,0],c2[:,1],c2[:,2],cam_poseM)
            sub1=self.cv_image1[max(corners[0].min(),0):min(corners[0].max(),self.cv_image1.shape[0]),
                                max(corners[1].min(),0):min(corners[1].max(),self.cv_image1.shape[1])]
            # sub2=self.cv_image2[max(corners[0].min(),0):min(corners[0].max(),self.cv_image1.shape[0]),max(corners[1].min(),0):min(corners[1].max(),self.cv_image1.shape[1])]
            # pdb.set_trace()
            val=calculate_blur(sub1)
        except Exception as e:
            print("Resulting cluster outside bounds of image - discarding")
            return False
        if val<20:
            # cv2.imshow("blur",sub1)
            # cv2.waitKey(1)
            return False
        return True

    # Re-usable function for creating clusters. Need to set internal prob variable first
    #   so that we can generate clusters above the specified threshold
    def create_clusters_from_depth(self, changed_dir, depth_subdir, image_name, threshold):
        # Create a mask of all points above threshold
        d_mask=(self.prob>threshold).astype(np.uint8)        
        xy=np.where(d_mask>0)
        try:
            # We need the top-level name of the project directory
            ln_s=changed_dir.split('/')
            for idx in range(1,len(ln_s)):
                if ln_s[-idx]!='':
                    project_dir=ln_s[-idx]
                    break
            # We also need the image number for determining the depth image
            image_number=image_name.split('_')[1]
            image_dir_and_name=[project_dir, image_name]

            # Create the depth image + point cloud
            if self.depth_image2 is None:
                self.depth_image2=self.load_depth_image(changed_dir+depth_subdir, image_number)
            pts, d_mask2=self.create_pcloud(image_dir_and_name, xy, self.depth_image2)

            # Apply clustering - do not save to local variable, in case that will
            #   mess up elsewhere
            clusters=self.apply_dbscan_clustering(pts,self.prob[xy][d_mask2],DEPTH_CLUSTER_EPS,self.cluster_min_count)

            return clusters

        except Exception as e:
            print("Clustering error: "+image_dir_and_name[0] + "/" + image_dir_and_name[1])
        return None

    # Nerf-Based change sets the prob variable using a difference between the raw + nerf images
    def set_nerf_based_change(self, nerf_outputs_dir, color_image_dir, image_name):
        self.cv_image1, clipV1=self.get_object_detection_data(nerf_outputs_dir+"/"+image_name)
        self.cv_image2, clipV2=self.get_object_detection_data(color_image_dir+"/"+image_name)
        if self.is_positive_change:
            self.prob=clipV2-clipV1
        else:
            self.prob=clipV1-clipV2

    # Alternative to NeRF - find a viewpoint that contains most of the same information
    #   and then build a 5 cm grid with the results. Change is measured as the delta
    #   between grids
    # def set_eg_based_change(self, 
    #                         initial_color_dir, changed_color_dir,
    #                         initial_depth_dir, changed_depth_dir,
    #                         image_name):
    #     print("EG: Load Files")
    #     # Load the new image first
    #     raw_image_name2=initial_color_dir+"/"+image_name
    #     image_number=image_name.split('.')[0].split('_')[1]
    #     self.cv_image2,clipV2=self.get_object_detection_data(raw_image_name2)
    #     cam_poseM2=self.images2.get_pose_by_name(image_dir_and_name[0],image_dir_and_name[1])
    #     self.depth_image2=self.load_depth_image(changed_depth_dir,image_number)

    #     print("EG: Select Pose")
    #     # Now select the best pose from the base set for the second image        
    #     xyz_ctr=np.vstack((get_center_point(self.depth_image2),1.0))
    #     tgt_obj=np.matmul(cam_poseM2,xyz_ctr)[:3].squeeze()
    #     best_key=self.images1.get_nearest_pose_by_angle(tgt_obj, cam_poseM2)

    #     print("EG: Load Base")
    #     # Load the base image - 
    #     image_dir_and_name1=self.images1.get_dir_and_name(best_key)
    #     raw_image_name1=self.color_image_name(image_dir_and_name1)
    #     self.cv_image1,clipV1=self.get_object_detection_data(raw_image_name1)
    #     depth_image1=self.load_depth_image(image_dir_and_name)

    #     print("EG: Create PCloud")
    #     # Build the set of points with nonzero depth and inference > 0.1
    #     rc2=np.where(clipV2>0.1)
    #     xyz2,d_mask2=self.create_pcloud(image_dir_and_name, rc2, self.depth_image2)       
    #     rc1=np.where(clipV1>0.1)
    #     xyz1,d_mask1=self.create_pcloud(image_dir_and_name1, rc1, depth_image1)       

    #     print("EG: Get Grid Boundaries")
    #     # Need the grid boundaries - need to base on actual pcloud rather than 
    #     #   just positive examples in order to support the negative condition
    #     xyz2_mn, xyz2_mx, num_dim=self.get_egrid_boundaries(image_dir_and_name, self.depth_image2)

    #     print("EG: Build EG")
    #     # Build an evidence grid based on the new image points
    #     #   then a second evidence grid with the same size as the new image
    #     #   can be constructed from the base data
    #     egrid2=max_grid3D(xyz2_mn, xyz2_mx, num_dim)
    #     egrid1=max_grid3D(xyz2_mn, xyz2_mx, num_dim)
    #     egridD=max_grid3D(xyz2_mn, xyz2_mx, num_dim)
    #     # add the points - only the highest score per cell is preserved
    #     egrid2.add_pts(xyz2,clipV2[rc2][d_mask2])
    #     egrid1.add_pts(xyz1,clipV1[rc1][d_mask1])

    #     print("EG: Delta")
    #     if self.is_positive_change:
    #         egridD.grid=egrid2.grid-egrid1.grid
    #     else:
    #         egridD.grid=egrid1.grid-egrid2.grid

    #     print("EG: Prob Matrix")
    #     #Last step - create the self.probability matrix, and
    #     #   fill in those points from earlier with updated values
    #     self.prob=np.zeros((clipV2.shape),dtype=float)
    #     rcD=np.where(self.depth_image2>0.1)
    #     xyzD,__=self.create_pcloud(image_dir_and_name, rcD, self.depth_image2)
    #     updated_values=egridD.get_values(xyzD)
    #     for idx in np.arange(rcD[0].shape[0]):
    #         self.prob[rcD[0][idx],rcD[1][idx]]=updated_values[idx]

    # # Visualize the high probability change regions
    def view_change_image(self, threshold, clusters_only=True):
        if self.is_positive_change:
            change_image=self.cv_image2
        else:
            change_image=self.cv_image1
        d_mask = (self.prob>threshold).astype(np.uint8)
        if clusters_only:
            d_mask = (self.prob>threshold).astype(np.uint8)
            return cv2.bitwise_and(change_image,change_image,mask=d_mask)
        else:
            d_maskN = (1-d_mask).astype(np.uint8)
            red_image=np.ones(change_image.shape,dtype=np.uint8)
            red_image[:,:,2]=255
            return cv2.bitwise_and(change_image,change_image,mask=d_maskN)+cv2.bitwise_and(red_image,red_image,mask=d_mask)

    # Search for change between one image and the baseline
    def detect_change(self, 
                      changed_dir,
                      outputs_subdir,
                      color_subdir,
                      depth_subdir,
                      image_name,
                      threshold:float=0.2):
        self.cv_image1=None
        self.cv_image2=None
        self.depth_image2=None
        try:
            # if self.use_nerf:
            self.set_nerf_based_change(changed_dir+outputs_subdir,changed_dir+color_subdir,image_name)
            # else:
            #     self.set_eg_based_change(image_dir_and_name)
        except Exception as e:
            print("Detect-Change, File load error")
            return False
        
        self.clusters=self.create_clusters_from_depth(changed_dir, depth_subdir, image_name, threshold)
        return (self.clusters is not None)
    
    # Re-run the last run, but with a different threshold
    def rerun_change_threshold(self, image_dir_and_name, threshold):
        self.clusters=self.create_clusters_from_depth(image_dir_and_name, threshold)
        return (self.clusters is not None)

    # Search for change between one image and the baseline
    def run_single_image_without_change(self, image_dir_and_name:list, threshold:float=0.2):
        self.cv_image1=None
        self.cv_image2=None
        self.depth_image2=None
        try:
            raw_image_name2=self.color_image_name(image_dir_and_name)
            self.cv_image2,self.prob=self.get_object_detection_data(raw_image_name2)
        except Exception as e:
            print("Single Image Detector, File load error: "+image_dir_and_name[0] + "/" + image_dir_and_name[1])
            return False
        self.clusters=self.create_clusters_from_depth(image_dir_and_name, threshold)
        return (self.clusters is not None)

    def create_initial_cluster_history(self, threshold:float, skip=4, save_file:str=None):
        if save_file is not None:
            try:
                with open(save_file, "rb") as fin:
                    cl_hist=pickle.load(fin)
                # If this detection target already exists in the cluster history
                #   then just load and exit
                if cl_hist is not None and self.detection_target in cl_hist.raw_clusters:
                    print("existing cluster history found - loading from file")
                    return cl_hist
            except Exception as e:
                print("No existing cluster history found - creating fresh")                
        cl_hist=cluster_history()
        plist=self.images1.get_pose_list()
        for idx, key in enumerate(plist):
            if idx % (skip+1)==0:
                try:
                    print("Processing: " + key)
                    image_dir_and_name=[self.images1.all_images[key]['directory'],self.images1.all_images[key]['name']]
                    raw_image_name=self.color_image_name(image_dir_and_name)
                    self.cv_image1, self.prob=self.get_object_detection_data(raw_image_name)
                    clusters=self.create_clusters_from_depth(image_dir_and_name, threshold)
                    if clusters is not None:
                        print("%s/%s: %d clusters"%(image_dir_and_name[0],image_dir_and_name[1],len(clusters)))
                        cl_hist.add_clusters(self.detection_target, clusters, key)
                    print("Clusters added")
                except Exception as e:
                    print("Failed to add history - continuing")
        if save_file is not None:
            try:
                with open(save_file, "wb") as fout:
                    pickle.dump(cl_hist,fout)
            except Exception as e:
                print("Error saving initial cluster history to file")
                    
        return cl_hist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('initial_dir',type=str,help='location of initial directory')
    parser.add_argument('changed_dir',type=str,help='location of changed directory')
    parser.add_argument('search_category',type=str,help='Prompt or object-type to use with the segmentation model')
    parser.add_argument('image_name',type=str,help='Investigate a particular image in the annotation file for change')
    parser.add_argument('--nerf_output_dir',type=str,default='renders',help='Path to the nerf output directory inside the changed directory (default=renders)')
    parser.add_argument('--initial_color_dir',type=str,default='rotated',help='Location of the raw color images within the initial directory (default = rotated)')
    parser.add_argument('--changed_color_dir',type=str,default='rotated',help='Location of the raw color images within the changed directory (default = rotated)')
    parser.add_argument('--changed_depth_dir',type=str,default=None,help='Location of depth images within the changed directory. By default, clusters are calculated without depth (default = None)')
    parser.add_argument('--initial_images_csv',type=str,default='images_geo.txt',help='Location of the images.csv file within the initial directory (default = images_geo.txt)')
    parser.add_argument('--changed_images_csv',type=str,default='images.txt',help='Location of the images.csv file within the changed directory (default = images_geo.txt)')
    parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
    # parser.add_argument('--nerf', action='store_true')
    # parser.add_argument('--no-nerf', dest='nerf', action='store_false')
    # parser.set_defaults(nerf=True)
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--yolo', dest='clip', action='store_false')
    parser.set_defaults(clip=True)
    parser.add_argument('--positive', action='store_true')
    parser.add_argument('--negative', dest='positive', action='store_false')
    parser.set_defaults(positive=True)
    args = parser.parse_args()

    eval_=image_comparator(args.initial_dir + "/" + args.initial_images_csv, 
                           args.changed_dir + "/" + args.changed_images_csv, 
                           args.search_category, args.positive, args.clip)

    if eval_.detect_change(args.changed_dir+"/",
                           args.nerf_output_dir,
                           args.changed_color_dir,
                           args.changed_depth_dir,
                           args.image_name,
                           args.threshold):
        change_image=eval_.view_change_image(args.threshold,clusters_only=False)
        plt.imshow(change_image)
        if args.clip:
            plt.imshow(change_image)
        else:
            plt.imshow(change_image[:,:,::-1])
        plt.show()
        pdb.set_trace()
    else:
        print("No change found -exiting")


