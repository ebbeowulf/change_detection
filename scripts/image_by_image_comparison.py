from image_set import image_set
import os
from PIL import Image
import pdb
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

ROOT_RAW_DIR="/data2/datasets/office/no_person/"
ROOT_NERF_DIR="/home/emartinso/projects/nerfstudio/renders/no_person_"

NERF_CLUSTER_EPS=5

def check_single_dim_overlap(a_min, a_max, b_min, b_max):
    if a_min<b_min:
        if a_max>b_min:
            return True
    elif a_max>b_min and a_min<b_max:
        return True
    return False

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

    def get_nerf_based_change(self, image_dir_and_name:list, threshold:float):
        raw_image=ROOT_RAW_DIR+image_dir_and_name[0]+"/rotated/" + image_dir_and_name[1]
        nerf_image=ROOT_NERF_DIR+image_dir_and_name[0] + "/" + image_dir_and_name[1]
        try:
            # print("1. Read images")
            cv_image1, clipV1=self.get_object_detection_data(nerf_image)
            cv_image2, clipV2=self.get_object_detection_data(raw_image)
            # print("2. Delta change")
            if self.is_positive_change:
                delta=clipV2-clipV1
                change_image=cv_image2
            else:
                delta=clipV1-clipV2
                change_image=cv_image1
            # print("3. Pt creation")
            d_mask=(delta>threshold).astype(np.uint8)                
            xy=np.where(d_mask>0)
            pts=np.vstack((xy[0],xy[1])).transpose()
            # print("4. Clustering")
            clusters=self.apply_dbscan_clustering(pts,delta[xy],NERF_CLUSTER_EPS)
            # print("5. Bitwise and")
            rgb_change=cv2.bitwise_and(change_image,change_image,mask=d_mask)
            return clusters, rgb_change
        except Exception as e:
            print("File load error: "+image_dir_and_name[0] + "/" + image_dir_and_name[1])
        return None, None

    def apply_dbscan_clustering(self, grid_pts, scores, eps):
        if grid_pts is None or grid_pts.shape[0]<5:
            return []
        CL2=DBSCAN(eps=eps, min_samples=5).fit(grid_pts,sample_weight=scores)
        clusters=[]
        for idx in range(10):
            whichP=np.where(CL2.labels_== idx)
            if len(whichP[0])<1:
                break
            clusters.append({'mean': grid_pts[whichP].mean(0),
                             'min': grid_pts[whichP].min(0),
                             'max': grid_pts[whichP].max(0),
                             'p_mean': scores[whichP].mean(),
                             'p_max': scores[whichP].max(),
                             'count': len(whichP[0])})
        return clusters

    def detect_change(self, image_dir_and_name:list, threshold:float):
        clusters, change_image=self.get_nerf_based_change(image_dir_and_name, threshold)
        return clusters, change_image
    
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
            clusters, change_image=self.detect_change(image_dir_and_name, 0.5)
            if clusters is None:
                continue
            if len(clusters)>0:
                changed[all_annotations[id]['path']]=clusters
                pdb.set_trace()
        pdb.set_trace()        

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_initial',type=str,help='location of images.csv for base dataset')
    parser.add_argument('images_changed',type=str,help='location of images.csv for changed dataset')
    parser.add_argument('search_category',type=str,help='Prompt or object-type to use with the segmentation model')
    parser.add_argument('annotation_file',type=str,help="Annotation file in COCO format")
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

        clusters, change_image = eval_.detect_change([directory,image_name], args.threshold)

        if change_image is None:
            print("No change found -exiting")
        else:

            plt.imshow(change_image)
            plt.show()
    else:
        eval_.evaluate_vs_gt(args.annotation_file)

