import os
import numpy as np
from change_pcloud_utils.colmap_utils import get_all_poses
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('colmap_dir',type=str,help='location of the directory containing the images.txt file (usually <nerf_dir>/colmap/sparse/0/)')
    parser.add_argument('--keyword',type=str,default=None, help='keyword to use in selecting images')
    args = parser.parse_args()

    poses=get_all_poses(args.colmap_dir,args.keyword)
    pose_np=np.array([ poses[key]['pose'] for key in poses.keys()])

    plt.plot(pose_np[:,0],pose_np[:,1])
    plt.show()

