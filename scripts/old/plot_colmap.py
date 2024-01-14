#!/usr/bin/env python3
import numpy as np
import pdb
import copy
import argparse
from image_set import read_image_csv
import matplotlib.pyplot as plt

def plot_path(P, M, color_path=[0,1,0], color_direction=[1,0,0]):
    forward=np.matmul(M,[0,0,0.3,1])
    for idx in range(len(P)):
        plt.plot([P[idx,0],forward[idx,0]],[P[idx,1],forward[idx,1]],color=color_direction)
    plt.plot(P[:,0],P[:,1],color=color_path)

def get_all_poses(all_images):
    # Create a dictionary to sort
    dct={}
    maxKey=0
    for im in all_images:
        dct[all_images[im]['id']]=im
        maxKey = max(maxKey, all_images[im]['id'])

    P=[]
    M=[]
    nm=[]
    for id in range(maxKey):
        try:
            if id in dct:
                im=all_images[dct[id]]
                P.append(im['global_pose'])
                M.append(im['global_poseM'])
                nm.append(im['name'])
        except Exception as e:
            pdb.set_trace()

    return np.array(P), np.array(M), nm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images',type=str,help='location of images.txt file to process')
    # parser.add_argument('images_sequence2',type=str,help='location of images in file system')
    args = parser.parse_args()

    all_images=read_image_csv(args.images)
    P1, M1, N1=get_all_poses(all_images)
    old_mask=[ 'frame' in x for x in N1 ]
    new_mask=[ 'change' in x for x in N1 ]

    plot_path(P1[old_mask,:],M1[old_mask,:,:],[0,1,0],[1,0,0])
    plot_path(P1[new_mask,:],M1[new_mask,:,:],[0,0,1],[0.5,0,0.5])
    plt.show()
