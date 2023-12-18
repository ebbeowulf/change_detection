from image_set import read_image_csv
import argparse
import matplotlib.pyplot as plt
import pdb
import numpy as np
import tf

def read_transforms(tf_file):
    import json
    with open(tf_file,'r') as fin:
        A=json.load(fin)
    
    dct={}
    maxKey=0
    for frame in A['frames']:
        dct[frame['colmap_im_id']]=frame
        maxKey = max(maxKey, frame['colmap_im_id'])
    
    P=[]
    M=[]
    nm=[]
    for id in range(maxKey):
        try:
            if id in dct:
                mat=np.array(dct[id]['transform_matrix'])
                P.append(mat[:3,3])
                M.append(mat)
                nm.append(dct[id]['file_path'])
        except Exception as e:
            pdb.set_trace()
    return np.array(P),M,nm

def get_all_poses(all_images):
    # Create a dictionary to sort
    dct={}
    maxKey=0
    for im in all_images:
        dct[im['id']]=im
        maxKey = max(maxKey, im['id'])

    P=[]
    Q=[]
    nm=[]
    for id in range(maxKey):
        try:
            if id in dct:
                P.append(dct[id]['trans'])
                Q.append(dct[id]['rot'])
                nm.append(dct[id]['name'])
        except Exception as e:
            pdb.set_trace()

    return np.array(P),np.array(Q),nm

def calculate_transform(colmap_matrix, pose_quat, pose_trans):
    pdb.set_trace()
    mm=tf.transformations.quaternion_matrix(pose_quat)
    mm[:3,3]=pose_trans
    T=np.matmul(mm,np.linalg.inv(colmap_matrix))
    return T

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('transforms1',type=str,help='location of transforms csv file to process')
    parser.add_argument('pose1',type=str,help='location of pose csv file to process')
    # parser.add_argument('images_sequence2',type=str,help='location of images in file system')
    args = parser.parse_args()

    # Image-csv file contains estimated pose from robot
    IS1=read_image_csv(args.pose1)
    P1i,Q1i,N1i=get_all_poses(IS1)
    # Transforms contains estimated pose from colmap
    P_colmap,M_colmap,N_colmap=read_transforms(args.transforms1)

    # Find the first colmap image in the pose sequence, 
    #   and calculate it's transform
    match_idx=np.where(np.array(N1i)==N_colmap[0])[0][0]
    T=calculate_transform(M_colmap[0], Q1i[match_idx], P1i[match_idx])
    M1=np.array([ np.matmul(T, mm) for mm in M_colmap ])
    P1=np.array([ mm[:3,3] for mm in M1 ])

    forward=np.matmul(M1,[0.3,0,0,1])
    pdb.set_trace()
    for idx in range(len(P1)):
        plt.plot([P1[idx,0],forward[idx,0]],[P1[idx,1],forward[idx,1]],color=[1,0,0])
    new_mask=[ 'change' in x for x in N_colmap ]

    plt.plot(P1[:,0],P1[:,1],color=[1,0,0])
    plt.plot(P1[new_mask,0],P1[new_mask,1])
    plt.show()

    pdb.set_trace()
    

