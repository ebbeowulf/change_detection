import argparse
import matplotlib.pyplot as plt
import pdb
import numpy as np
import tf
from image_set import image_set, create_image_vector

FILTER_SIZE=5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_initial',type=str,help='location of initial pose csv file to process')
    parser.add_argument('clip_initial',type=str,help='initial clip csv file to process')
    parser.add_argument('images_change',type=str,help='location of change pose csv file to process')
    parser.add_argument('clip_change',type=str,help='change clip csv file to process')
    args = parser.parse_args()

    P_initial=image_set(args.images_initial,1.5)
    P_change=image_set(args.images_change,1.5)
    clip_initial=create_image_vector(args.clip_initial)
    clip_change=create_image_vector(args.clip_change)

    for idx in range(1,len(P_change.all_images),FILTER_SIZE):
        #Get the target pose
        tgt_pose=P_change.get_pose_by_id(idx)
        if tgt_pose is None:
            continue
        
        # Get the lists of related poses for both the initial
        #   and change conditions
        rel_initial=P_initial.get_related_poses(tgt_pose)
        rel_change=P_change.get_related_poses(tgt_pose)

        #Build the associated vector
        clipV_arr_initial=clip_initial.get_array(rel_initial)
        clipV_arr_change=clip_change.get_array(rel_change)

        # Simplify vectors
        V_initial=clipV_arr_initial.sum(1)/len(rel_initial)
        V_change=clipV_arr_change.sum(1)/len(rel_change)
        
        pdb.set_trace()