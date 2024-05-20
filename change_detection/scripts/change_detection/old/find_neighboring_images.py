import argparse
import pdb
import numpy as np
from image_set import image_set, create_image_vector
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('depthP_csv',type=str,help='location of depth_csv file with database of image poses + extracted center points')
parser.add_argument('starter_image',default=None,type=str,help='load images that are nearby to the indicated image')
parser.add_argument('--max-distance',default=1.0,type=float,help='max dissance between target pt and recorded image')
parser.add_argument('--image-dir',type=str,default=None,help='location of raw images - only use to visualize the results')
args = parser.parse_args()

GI=image_set(args.depthP_csv)
tgt_pose=GI.get_pose_by_name(args.starter_image)
im_list=GI.get_related_poses(tgt_pose,args.max_distance)
name_list=[]
for im in im_list:
    print(im)

if args.image_dir is not None:
    tgt_fName=args.image_dir+'/'+args.starter_image
    rgb_tgt=cv2.imread(tgt_fName)
    pdb.set_trace()
    cv2.imshow("tgt",rgb_tgt)
    rr=np.random.permutation(range(len(im_list)))
    for idx in rr:
        compare_fName=args.image_dir+'/'+im_list[idx]
        print(compare_fName)
        rgb_compare=cv2.imread(compare_fName)
        cv2.imshow("compare",rgb_compare)
        cv2.waitKey(0)

