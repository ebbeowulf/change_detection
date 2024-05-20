import numpy as np
import argparse
from image_set import get_neighboring_images, create_image_vector
import pdb
from grid import average_grid
import matplotlib.pyplot as plt

class build_object_map():
    def __init__(self, image_csv:str, clip_csv:str):
        self.GI=get_neighboring_images(image_csv)
        self.CV=create_image_vector(clip_csv)
        arr = self.GI.get_all_poses()
        mn=arr.min(0)-1.5
        mx=arr.max(0)+1.5
        dimensions = np.ceil(mx-mn).astype(int)
        self.labels=self.CV.get_labels()
        self.grid=average_grid(mn[0],mn[1],mx[0],mx[1],[dimensions[0],dimensions[1]])

    def add_score(self, position, clip_res):
        self.grid.add_value(position[0], position[1], clip_res)
    
    def tally_run(self):
        for im in self.GI.all_images:
            print(im['name'])
            tgt_pose=self.GI.get_pose_by_name(im['name'])
            im_list=self.GI.get_related_poses(tgt_pose,2.0, 0.5)
            name_list=[]
            for im in im_list:
                name_list.append(im['name'])
            arr=self.CV.get_array(name_list)
            self.add_score(tgt_pose, arr.mean(1))
        return self.grid.get_grid_average()
         


parser=argparse.ArgumentParser()
parser.add_argument('images_csv',type=str,help='location of images.txt file with database of image poses')
parser.add_argument('clip_csv',type=str,help='location of clip.csv file with database of clip scores for each image')
args = parser.parse_args()

omap=build_object_map(args.images_csv, args.clip_csv)
grid=omap.tally_run()
pdb.set_trace()