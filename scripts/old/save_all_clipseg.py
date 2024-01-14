from accumulate_evidence import map_run

from scipy.spatial.transform import Rotation as R
from image_set import image_set
import argparse
from points_from_depth import read_image_csv
import numpy as np
import cv2

prompts=["a computer", 
         "a suspicious object", 
         "signs of a break-in", 
         "a package",
         "a mess"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_initial',type=str,help='location of initial image txt file')
    parser.add_argument('color_dir',type=str,help='initial clip csv file to process')
    parser.add_argument('depth_dir',type=str,help='location of change pose csv file to process')
    parser.add_argument('save_dir',type=str,help='where to save the results')
    parser.add_argument('--num_images',type=int,default=1000, help='number of images to process (default=100)')
    args = parser.parse_args()

    MR=map_run(prompts, args.color_dir, args.depth_dir)
    all_images=read_image_csv(args.images_initial)

    names=[ im for im in all_images.keys() ]
    np.random.shuffle(names)

    for idx,im in enumerate(names[:args.num_images]):
        print(str(idx)+": " + im)
        rotM=all_images[im]['global_poseM']
        MR.process_and_add_image(im, rotM, save_dir=args.save_dir)

    MI_image=MR.generate_max_height_image()
    cv2.imshow("max height",MI_image)
    cv2.waitKey(0)


