import argparse
import matplotlib.pyplot as plt
import pdb
import numpy as np
import tf
from image_set import image_set, create_image_vector

FILTER_SIZE=5
BASE_OBJECTS=['cabinet', 'carpet', 'wood', 'wall', 'tile', 'linoleum', 'floor', 'desk', 'table']
TGT_OBJECTS=['printer', 'tv', 'monitor', 'paper', 'binder', 'painting', 'poster', 'book', 'keyboard', 'laptop', 'robot', 'cart', 'flag', 'fire extinguisher']

def gaussian_pdf_std(mean, stdev, V):
    return np.exp(-0.5*np.power((mean[:]-V)/stdev,2))

def gaussian_pdf_multi(mean, inverse_cov, V):
    deltaV=mean-V
    A1=np.matmul(deltaV.transpose(),inverse_cov)
    return float(np.exp(-0.5*np.matmul(A1,deltaV)))

class estimate_change():
    def __init__(self, images_initial, clip_initial, images_change, clip_change):
        self.P_initial=image_set(images_initial)
        self.P_change=image_set(images_change)
        self.clip_initial=create_image_vector(clip_initial)
        self.clip_change=create_image_vector(clip_change)
        # self.clip_initial.set_base_label_mask(BASE_OBJECTS)
        # self.clip_change.set_base_label_mask(BASE_OBJECTS)
        self.clip_initial.set_label_mask(TGT_OBJECTS)
        self.clip_change.set_label_mask(TGT_OBJECTS)

    def prep_change_detect(self,tgt_pose):
        rel_initial=self.P_initial.get_related_poses(tgt_pose, 1.0, 0.5)
        rel_change=self.P_change.get_related_poses(tgt_pose, 1.0, 0.5)

        object_model=self.clip_initial.create_gaussian_model(rel_initial,False)
        # base_model=self.clip_initial.create_gaussian_model(rel_initial,True)

        # if object_model is None or base_model is None:
        if object_model is None:
            return None

        object_arr_change=self.clip_change.get_array(rel_change)
        # base_arr_change=self.clip_change.get_array(rel_change,True)

        return {'rel_initial': rel_initial, 
                'rel_change': rel_change, 
                'object_model': object_model,
                # 'base_model': base_model,
                'object_arr_change': object_arr_change}
                # 'base_arr_change': base_arr_change}

    def score_change(self, model, vector_arr):
        scores=np.zeros((vector_arr.shape[1]),dtype=float)
        for idx in range(vector_arr.shape[1]):
            scores[idx]=np.log(gaussian_pdf_multi(model['mean'][:,0], model['inv_cov'], vector_arr[:,idx])+1e-6)
        return scores

    def build_change_vector(self, details):
        objS=self.score_change(details['object_model'],details['object_arr_change'])
        # baseS=self.score_change(details['base_model'],details['base_arr_change'])
        deltaV=details['object_arr_change']-details['object_model']['mean']
        # plt.bar(self.clip_change.get_labels(),deltaV.mean(1))
        return deltaV.mean(1)

    def query_by_image_id(self, idx):
        tgt_pose=self.P_change.get_pose_by_id(idx)
        if tgt_pose is None:
            return None

        details=self.prep_change_detect(tgt_pose)
        if details is None:
            return None
        return self.build_change_vector(details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_initial',type=str,help='location of initial pose csv file to process')
    parser.add_argument('clip_initial',type=str,help='initial clip csv file to process')
    parser.add_argument('images_change',type=str,help='location of change pose csv file to process')
    parser.add_argument('clip_change',type=str,help='change clip csv file to process')
    args = parser.parse_args()

    estCh=estimate_change(args.images_initial, args.clip_initial, args.images_change, args.clip_change)

    all_scores=[]
    valid_im=[]
    for idx in range(1,1000,FILTER_SIZE):        
        score=estCh.query_by_image_id(idx)
        if score is not None:
            all_scores.append(score)
            valid_im.append(idx)
        # all_scores.append([idx,score[0],score[1],score[2]])
        # #Get the target pose
        # tgt_pose=P_change.get_pose_by_id(idx)
        # if tgt_pose is None:
        #     continue
        
        # # Get the lists of related poses for both the initial
        # #   and change conditions
        # rel_initial=P_initial.get_related_poses(tgt_pose)
        # rel_change=P_change.get_related_poses(tgt_pose)

        # object_model=clip_initial.create_gaussian_model(rel_initial,False)
        # base_model=clip_initial.create_gaussian_model(rel_initial,True)

        # score=[0,0,0]
        # for im in rel_change:
        #     try:
        #         objectV=clip_change.get_vector(im,False)
        #         baseV=clip_change.get_vector(im,True)
        #         if objectV is None or baseV is None:
        #             continue
        #         objectS=np.log(gaussian_pdf_multi(object_model['mean'], object_model['inv_cov'], objectV)+1e-6)
        #         baseS=np.log(gaussian_pdf_multi(base_model['mean'], base_model['inv_cov'], baseV)+1e-6)
        #         score[0]+=objectS
        #         score[1]+=baseS
        #         score[2]+=objectS-baseS
        #     except Exception as e:
        #         pdb.set_trace()
    pdb.set_trace()
    AS=np.array(all_scores)
    plt.plot(valid_im, AS.sum(1))
    plt.plot(AS[:,0],AS[:,1])
    plt.plot(AS[:,0],AS[:,2])
    plt.plot(AS[:,0],AS[:,3])
    # plt.savefig("mygraph.png")
    plt.show()
    pdb.set_trace()