import argparse
import matplotlib.pyplot as plt
import pdb
import numpy as np
import tf
from image_set import image_set, create_image_vector

FILTER_SIZE=5
BASE_OBJECTS=['cabinet', 'carpet', 'wood', 'wall', 'tile', 'linoleum', 'floor', 'desk', 'table']

def gaussian_pdf_std(mean, stdev, V):
    return np.exp(-0.5*np.power((mean[:]-V)/stdev,2))

def gaussian_pdf_multi(mean, inverse_cov, V):
    deltaV=mean-V
    A1=np.matmul(deltaV.transpose(),inverse_cov)
    return float(np.exp(-0.5*np.matmul(A1,deltaV)))

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
    clip_initial.set_base_label_mask(BASE_OBJECTS)
    clip_change.set_base_label_mask(BASE_OBJECTS)

    all_scores=[]
    for idx in range(1,len(P_change.all_images),FILTER_SIZE):
        #Get the target pose
        tgt_pose=P_change.get_pose_by_id(idx)
        if tgt_pose is None:
            continue
        
        # Get the lists of related poses for both the initial
        #   and change conditions
        rel_initial=P_initial.get_related_poses(tgt_pose)
        rel_change=P_change.get_related_poses(tgt_pose)

        object_model=clip_initial.create_gaussian_model(rel_initial,False)
        base_model=clip_initial.create_gaussian_model(rel_initial,True)

        score=[0,0,0]
        for im in rel_change:
            try:
                objectV=clip_change.get_vector(im,False)
                baseV=clip_change.get_vector(im,True)
                if objectV is None or baseV is None:
                    continue
                objectS=np.log(gaussian_pdf_multi(object_model['mean'], object_model['inv_cov'], objectV)+1e-6)
                baseS=np.log(gaussian_pdf_multi(base_model['mean'], base_model['inv_cov'], baseV)+1e-6)
                score[0]+=objectS
                score[1]+=baseS
                score[2]+=objectS-baseS
            except Exception as e:
                pdb.set_trace()
        all_scores.append([idx,score[0],score[1],score[2]])
    AS=np.array(all_scores)
    plt.plot(AS[:,0],AS[:,1])
    plt.plot(AS[:,0],AS[:,2])
    plt.plot(AS[:,0],AS[:,3])
    plt.savefig("mygraph.png")
    pdb.set_trace()