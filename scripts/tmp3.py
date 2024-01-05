import numpy as np
from accumulate_evidence import map_run
import pickle
import cv2
import pdb

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

image_num="00670"
# pdb.set_trace()
color_fName1, poseM1, clipV1, prompts1=load_inference('/data2/datasets/office/no_person/monitor/side_by_side/initial/rgb_' + image_num + '.png.pkl')
color_fName2, poseM2, clipV2, prompts2=load_inference('/data2/datasets/office/no_person/monitor/side_by_side/current/rgb_'+ image_num + '.png.pkl')
color_image1=cv2.imread('/data2/datasets/office/no_person/monitor/side_by_side/initial/rgb_' + image_num + '.png',-1)
depth_image1=cv2.imread('/data2/datasets/office/no_person/monitor/side_by_side/initial/depth_' + image_num + '.png',-1)
color_image2=cv2.imread('/data2/datasets/office/no_person/monitor/side_by_side/current/rgb_' + image_num + '.png',-1)
depth_image2=cv2.imread('/data2/datasets/office/no_person/monitor/depth/depth_' + image_num + '.png',-1)
for i in range(clipV1.shape[0]):
    print(prompts1[i])
    delta=clipV2[i]-clipV1[i]
    res=cv2.bitwise_and(color_image1,color_image1,mask=(np.abs(delta)>0.5).astype(np.uint8))
    cv2.imshow("delta",res)
    cv2.waitKey(0)
pdb.set_trace()