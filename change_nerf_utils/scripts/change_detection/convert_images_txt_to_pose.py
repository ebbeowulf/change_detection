import argparse
import os
import numpy as np
import pdb
import json
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_txt',type=str,help='location of images.txt file produced by colmap model_converter')
    args = parser.parse_args()

    # Load the location csv
    with open(args.images_txt, 'r') as fin:
        A=fin.readlines()

    # Remove comments at beginning
    start_line=-1
    for ln_idx,ln in enumerate(A):
        if ln[0]=='#':
            continue
        start_line=ln_idx
        break

    # Need to skip every other line while parsing
    for ln_idx in range(start_line,len(A),2):
        if A[ln_idx][-1]=='\n':
            ln=A[ln_idx][:-1]
        else:
            ln=A[ln_idx]
        vals=ln.split(' ')
        quat_xyzw=[float(vals[2]),float(vals[3]),float(vals[4]),float(vals[1])]
        rot=R.from_quat(quat_xyzw) #this is an inverted matrix pointing back towards the center of the camera
        trans=np.array([float(vals[5]),float(vals[6]),float(vals[7])])
        # According to the NerfStudio documentation, pose is recovered by -rot^t * T
        #   https://colmap.github.io/format.html
        rot_matrix=rot.as_matrix()
        pose=np.matmul(-rot_matrix.transpose(),trans)
        print(f"{vals[-1]} {pose[0]} {pose[1]} {pose[2]}")

