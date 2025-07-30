import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import pdb

## Retrieve the camera poses
##      Need two files for this purpose.
##      images.txt - contains the actual image locations as
##      dataparsar_transform.json - 
def get_all_poses(colmap_dir, keyword:str):
    # Load the location csv
    try:
        with open(colmap_dir+"/images.txt", 'r') as fin:
            A=fin.readlines()
    except Exception as e:
        print("Failed to open images.txt in " + colmap_dir + " - exiting")
        os.exit(-1)
        
    # Remove comments at beginning
    start_line=-1
    for ln_idx,ln in enumerate(A):
        if ln[0]=='#':
            continue
        start_line=ln_idx
        break

    all_poses=dict()
    for ln_idx in range(start_line,len(A),2):
        if A[ln_idx][-1]=='\n':
            ln=A[ln_idx][:-1]
        else:
            ln=A[ln_idx]
        vals=ln.split(' ')
        if keyword is None or keyword in vals[-1]:
            quat_xyzw=[float(vals[2]),float(vals[3]),float(vals[4]),float(vals[1])]
            rot=R.from_quat(quat_xyzw) #this is an inverted matrix pointing back towards the center of the camera
            trans=np.array([float(vals[5]),float(vals[6]),float(vals[7])])
            # According to the NerfStudio documentation, pose is recovered by -rot^t * T
            #   https://colmap.github.io/format.html
            rot_matrix=rot.as_matrix()
            rot_matrix_c2w=rot_matrix.transpose()
            # This hack works - but don't use as it will probably break later
            pose=np.matmul(-rot_matrix_c2w,trans)
            all_poses[vals[-1]]={'pose': pose, 'rot_cam2world': rot_matrix.transpose()}
    return all_poses

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('colmap_dir',type=str,help='location of the directory containing the images.txt file (usually <nerf_dir>/colmap/sparse/0/)')
    parser.add_argument('--keyword',type=str,default=None, help='keyword to use in selecting images')
    args = parser.parse_args()

    poses=get_all_poses(args.colmap_dir,args.keyword)

    pose_np=np.array([ poses[key]['pose'] for key in poses.keys()])
    import matplotlib.pyplot as plt
    plt.plot(pose_np[:,0],pose_np[:,1])
