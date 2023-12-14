import argparse
import pdb
import numpy as np
from image_set import get_neighboring_images, create_image_vector
    
parser=argparse.ArgumentParser()
parser.add_argument('images_txt',type=str,help='location of images.txt file with database of image poses')
parser.add_argument('--starter-image',default=None,type=str,help='load images that are nearby to the indicated image')
parser.add_argument('--max-distance',default=2.0,type=float,help='max ditsance between target pt and recorded image')
parser.add_argument('--max-angle',default=0.5,type=float,help='max angle between target pt and recorded image')
parser.add_argument('--clip-csv',default=None,type=str,help='optional clip csv file to create combined vector')
args = parser.parse_args()

GI=get_neighboring_images(args.images_txt)
tgt_pose=GI.get_pose_by_name(args.starter_image)
im_list=GI.get_related_poses(tgt_pose,args.max_distance,args.max_angle)
name_list=[]
for im in im_list:
    print("%s: [%0.2f,%0.2f,%0.2f]"%(im['name'],im['trans'][0],im['trans'][1],im['trans'][2]))
    name_list.append(im['name'])

if args.clip_csv is not None:
    CV=create_image_vector(args.clip_csv)
    arr=CV.get_array(name_list)
    pdb.set_trace()
