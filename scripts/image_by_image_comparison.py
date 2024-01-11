from image_set import image_set
import os
from PIL import Image
import pdb
import argparse

ROOT_RAW_DIR="/data2/datasets/office/no_person/"
ROOT_NERF_DIR="/home/emartinso/projects/nerfstudio/renders/no_person_"

class image_comparator():
    def __init__(self, 
                 images_initial_csv:str, 
                 images_secondary_csv:str, 
                 detection_target: str,
                 use_nerf:bool, 
                 use_clip:bool):
        self.use_nerf=use_nerf
        self.detection_target=detection_target
        if use_clip:
            from clip_segmentation import clip_seg
            self.detect=clip_seg([self.detection_target])
        else:
            from yolo_segmentation import yolo_segmentation
            self.detect=yolo_segmentation()
        self.images1=image_set(images_initial_csv)
        self.images2=image_set(images_secondary_csv)
    
    def get_nerf_based_change(self, image_filename, is_positive_change:bool):
        pdb.set_trace()
        raw_image=ROOT_RAW_DIR+image_filename
        nerf_image=ROOT_NERF_DIR+image_filename
        try:
            clipV1=self.detect.process_file(nerf_image)
            clipV2=self.detect.process_file(raw_image)
            if is_positive_change:
                return clipV2-clipV1
            else:
                return clipV1-clipV2
        except Exception as e:
            print("File load error: "+image_filename)
        return None

    def detect_change(self, image_filename, is_positive_change:bool, threshold:float):
        change=self.get_nerf_based_change(image_filename, is_positive_change)
        pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_initial',type=str,help='location of images.csv for base dataset')
    parser.add_argument('images_changed',type=str,help='location of images.csv for changed dataset')
    parser.add_argument('search_category',type=str,help='Prompt or object-type to use with the segmentation model')
    parser.add_argument('image_name',type=str,help='Which image to investigate for change')
    parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ')
    parser.add_argument('--nerf', action='store_true')
    parser.add_argument('--no-nerf', dest='nerf', action='store_false')
    parser.set_defaults(nerf=True)
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--no-clip', dest='clip', action='store_false')
    parser.set_defaults(clip=True)
    parser.add_argument('--positive', action='store_true')
    parser.add_argument('--negative', dest='positive', action='store_false')
    parser.set_defaults(positive=True)
    args = parser.parse_args()

    eval_=image_comparator(args.images_initial, args.images_changed, args.search_category, args.nerf, args.clip)
    eval_.detect_change(args.image_name, args.positive, args.threshold)


