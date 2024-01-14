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

ROOT_RAW_DIR="/data2/datasets/office/no_person/"
ROOT_NERF_DIR="/home/emartinso/projects/nerfstudio/renders/no_person_"

# DBSCAN Parameters
DEPTH_CLUSTER_EPS=0.1          
DEPTH_CLUSTER_MINC=1000
MAX_CLUSTERING_SAMPLES=50000 #DBSCAN struggles to handle more points than this, so randomly sample after this threshold

K_ROTATED=[906.7647705078125, 0.0, 368.2167053222656, 
                0.0, 906.78173828125, 650.24609375,                 
                0.0, 0.0, 1.0]
F_X=K_ROTATED[0]
C_X=K_ROTATED[2]
F_Y=K_ROTATED[4]
C_Y=K_ROTATED[5]

def get_3D_point(x_pixel, y_pixel, depth):
    x = (x_pixel - C_X) * depth / F_X
    y = (y_pixel - C_Y) * depth / F_Y
    return np.vstack((x,y,depth))

class image_comparator():
    def __init__(self, 
                 images_initial_csv:str, 
                 images_secondary_csv:str, 
                 detection_target: str,
                 is_positive_change:bool,
                 use_nerf:bool, 
                 use_clip:bool):
        self.use_nerf=use_nerf
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
    
    def get_object_detection_data(self, image):
        cv_image=self.detect.process_file(image)
        if self.detect_id is None:
            self.detect_id=self.detect.get_id(self.detection_target)
        clipV=self.detect.get_prob_array(self.detect_id)
        if clipV is None:
            clipV=np.zeros(cv_image.shape[0:2],dtype=float)
        return cv_image, clipV

    def create_pcloud(self, image_dir_and_name:list, pts_xy):
        number=image_dir_and_name[1].split('_')[1]
        # Load the depth image
        depth_image_name=ROOT_RAW_DIR+image_dir_and_name[0]+"/depth/depth_" + number
        depth_image = cv2.imread(depth_image_name, -1)
        depth_image=cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
        # Get the untransformed points
        D=depth_image[pts_xy]
        d_mask=np.where(D>100) # filter out the zero depth points
        pts=get_3D_point(pts_xy[0][d_mask],pts_xy[1][d_mask],D[d_mask]/1000.0)                
        # Transform to map space using the image calibration info
        pts=np.vstack((pts,np.ones((1,pts.shape[1]),dtype=float)))
        cam_poseM=self.images2.get_pose_by_name(image_dir_and_name[0],image_dir_and_name[1])
        pts=np.matmul(cam_poseM,pts)[:3,:].transpose()
        return pts, d_mask, depth_image

    def get_nerf_based_change(self, image_dir_and_name:list, threshold:float):
        raw_image=ROOT_RAW_DIR+image_dir_and_name[0]+"/rotated/" + image_dir_and_name[1]
        nerf_image=ROOT_NERF_DIR+image_dir_and_name[0] + "/" + image_dir_and_name[1]
        try:
            # print("1. Read images")
            self.cv_image1, clipV1=self.get_object_detection_data(nerf_image)
            self.cv_image2, clipV2=self.get_object_detection_data(raw_image)

            # print("2. Delta change")
            if self.is_positive_change:
                self.delta=clipV2-clipV1
            else:
                self.delta=clipV1-clipV2
            print("3. Pt creation")
            d_mask=(self.delta>threshold).astype(np.uint8)                
            xy=np.where(d_mask>0)
            pts, d_mask2, self.depth_image=self.create_pcloud(image_dir_and_name, xy)

            print("4. Clustering")
            self.clusters=self.apply_dbscan_clustering(pts,self.delta[xy][d_mask2],DEPTH_CLUSTER_EPS,DEPTH_CLUSTER_MINC)

            return True
        except Exception as e:
            print("File load error: "+image_dir_and_name[0] + "/" + image_dir_and_name[1])
        return False

    def view_change_image(self, threshold):
        if self.is_positive_change:
            change_image=self.cv_image2
        else:
            change_image=self.cv_image1
        d_mask = (self.delta>threshold).astype(np.uint8)
        return cv2.bitwise_and(change_image,change_image,mask=d_mask)

    def sample_pts(self, grid_pts, scores):
        if grid_pts.shape[0]<MAX_CLUSTERING_SAMPLES:
            return grid_pts, scores
        
        rr=np.random.choice(np.arange(grid_pts.shape[0]),size=MAX_CLUSTERING_SAMPLES)
        return grid_pts[rr], scores[rr]
    
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

    def detect_change(self, image_dir_and_name:list, threshold:float=0.2):
        return self.get_nerf_based_change(image_dir_and_name, threshold)
    
    def build_cluster_history(self, threshold, skip=4, existing_cl_hist:cluster_history=None):
        plist=self.images2.get_pose_list()
        if existing_cl_hist is not None:
            cl_hist=existing_cl_hist
            cl_hist.setup_category(self.detection_target)
        else:
            cl_hist=cluster_history()
        for idx, key in enumerate(plist):
            if idx % (skip+1)==0:
                print("Processing: " + key)
                image_dir_and_name=[self.images2.all_images[key]['directory'],self.images2.all_images[key]['name']]
                if self.detect_change(image_dir_and_name, threshold):
                    print("%s/%s: %d clusters"%(image_dir_and_name[0],image_dir_and_name[1],len(self.clusters)))
                    cl_hist.add_clusters(self.detection_target, self.clusters, key)
        return cl_hist
    
    def evaluate_vs_gt(self, gt_file):
        with open(gt_file,'r') as fin:
            import json
            A=json.load(fin)
        all_annotations={img['id']: {'path': img['file_name'], 
                                     'scenario': img['file_name'].split('/')[0],
                                     'name': img['file_name'].split('/')[1],
                                     'boxes': []} for img in A['images']}
        for annot_ in A['annotations']:
            all_annotations[annot_['image_id']]['boxes'].append(annot_)
        changed={}
        for id in all_annotations:
            image_dir_and_name=[all_annotations[id]['scenario'],all_annotations[id]['name']]
            if self.detect_change(image_dir_and_name):
                if len(self.clusters)>0:
                    changed[all_annotations[id]['path']]=self.clusters
        pdb.set_trace()        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_initial',type=str,help='location of images.csv for base dataset')
    parser.add_argument('images_changed',type=str,help='location of images.csv for changed dataset')
    parser.add_argument('search_category',type=str,help='Prompt or object-type to use with the segmentation model')
    parser.add_argument('--annotation-file',type=str,default=None,help="Annotation file in COCO format (optional)")
    parser.add_argument('--image-name',type=str,default=None, help='Investigate a particular image in the annotation file for change')
    parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
    parser.add_argument('--nerf', action='store_true')
    parser.add_argument('--no-nerf', dest='nerf', action='store_false')
    parser.set_defaults(nerf=True)
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--yolo', dest='clip', action='store_false')
    parser.set_defaults(clip=True)
    parser.add_argument('--positive', action='store_true')
    parser.add_argument('--negative', dest='positive', action='store_false')
    parser.set_defaults(positive=True)
    args = parser.parse_args()

    eval_=image_comparator(args.images_initial, args.images_changed, args.search_category, args.positive, args.nerf, args.clip)

    if args.image_name is not None:
        ln_s=args.image_name.split('/')
        if len(ln_s)>1:
            directory=ln_s[0]
            image_name=ln_s[1]
        else:
            directory=None
            image_name=args.image_name

        if eval_.detect_change([directory,image_name], args.threshold):
            change_image=eval_.view_change_image(args.threshold)
            plt.imshow(change_image)
            plt.show()
        else:
            print("No change found -exiting")

    elif args.annotation_file is not None:
        eval_.evaluate_vs_gt(args.annotation_file)

