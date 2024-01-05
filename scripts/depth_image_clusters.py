from points_from_depth import points_from_depth
import argparse
import numpy as np
import pickle
import pdb
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

K_rotated=[906.7647705078125, 0.0, 368.2167053222656, 
                0.0, 906.78173828125, 650.24609375,                 
                0.0, 0.0, 1.0]
F_x=K_rotated[0]
C_x=K_rotated[2]
F_y=K_rotated[4]
C_y=K_rotated[5]
prob_threshold=0.6

def get_3D_point(x_pixel, y_pixel, depth):
    x = (x_pixel - C_x) * depth / F_x
    y = (y_pixel - C_y) * depth / F_y
    return [x,y,depth]

def get_clusters(clipV, depth_image, threshold=0.5):
    clusters_raw=[]
    clusters_stats=[]
    # skip the last category in this instance - the wooden desk takes too long
    for idx in range(clipV.shape[0]-1):
        clusters_raw.append([])
        clusters_stats.append([])
        positive=np.where(clipV[idx]>threshold)
        if len(positive[0])<5:
            continue
        pts=get_3D_point(positive[1],positive[0],depth_image[positive])
        pts=np.array(pts).transpose()
        prob=clipV[idx][positive]
        pos_depth=(pts[:,2]>0.3)
        pts=pts[pos_depth]
        prob=prob[pos_depth]
        if pts.shape[0]<5:
            continue
        CL2=DBSCAN(eps=0.2, min_samples=5).fit(pts,sample_weight=prob)
        for cl_idx in range(100):
            whichP=np.where(CL2.labels_==cl_idx)
            if len(whichP[0])>0:
                # p_log_odds=
                cl_stats={'count': len(whichP[0]), 'mean': pts[whichP].mean(0), 'std': pts[whichP].std(0), 'max': pts[whichP].max(0), 'min': pts[whichP].min(0), 'p_mean': prob[whichP].mean(), 'p_max': prob[whichP].max()}
                cl_raw={'pts': pts[whichP], 'prob': prob[whichP]}
                clusters_stats[idx].append(cl_stats)
                clusters_raw[idx].append(cl_raw)
            else:
                break
    return clusters_stats, clusters_raw

def get_depth_image(color_fName, depth_dir):
    # Get the depth image
    color_fName_s=color_fName.split('_')
    depth_fName=depth_dir+"/depth_"+color_fName_s[-1]
    depth_image=cv2.imread(depth_fName,cv2.IMREAD_UNCHANGED)
    depth_image=cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
    return depth_image/1000.0

def load_inference(pickle_file):
    try:
        with open(pickle_file, "rb") as fin:
            clipV=pickle.load(fin)
            global_poseM=pickle.load(fin)
            prompts=pickle.load(fin)
            color_fName=pickle.load(fin)
        return color_fName, global_poseM, clipV, prompts
    except Exception as e:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_image_dir',type=str,help='location of the depth images')
    parser.add_argument('numpy_dir', type=str, help='location of numpy files')
    parser.add_argument('--numpy_prefix', type=str, default='new_rgb', help='prefix for numpy files containing the clip_seg results')
    parser.add_argument('--save_file', type=str, default='depth_clusters', help='save file')
    args = parser.parse_args()

    files=os.listdir(args.numpy_dir)
    count=0
    all_stats={}
    all_raw={}
    for file_ in files:
        if 'pkl' in file_ and args.numpy_prefix in file_:
            count+=1
            print(file_)
            color_fName, global_poseM, clipV, prompts = load_inference(args.numpy_dir + "/" + file_)
            depthI=get_depth_image(color_fName, args.depth_image_dir)
            stats, raw=get_clusters(clipV, depthI, threshold=prob_threshold)
            all_stats[color_fName]={'pose': global_poseM, 'stats': stats}
            all_raw[color_fName]={'pose': global_poseM, 'stats': raw}

            if count%10==0:
                with open(args.numpy_dir + "/" + args.save_file + ".stats.pkl","wb") as fout:
                    pickle.dump(all_stats, fout)
                with open(args.numpy_dir + "/" + args.save_file + ".raw.pkl","wb") as fout:
                    pickle.dump(all_raw, fout)                  

    # dump remaining regardless
    with open(args.numpy_dir + "/" + args.save_file + ".stats.pkl","wb") as fout:
        pickle.dump(all_stats,fout)
    with open(args.numpy_dir + "/" + args.save_file + ".raw.pkl","wb") as fout:
        pickle.dump(all_raw,fout)                  
    