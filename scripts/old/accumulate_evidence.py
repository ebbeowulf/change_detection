from PIL import Image
import pdb
import cv2
import numpy as np
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from grid import evidence_grid3D
import pickle

class clip_seg():
    #from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb
    def __init__(self, prompts):
        print("Reading model")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.prompts=prompts
    
    def sigmoid(self, arr):
        return (1.0/(1.0+np.exp(-arr)))
    
    def process_image(self, image):
        print("Inference")
        inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding="max_length", return_tensors="pt")
        # predict
        with torch.no_grad():
            outputs = self.model(**inputs)

        if len(self.prompts)>1:
            preds = outputs.logits.unsqueeze(1)
            P2=self.sigmoid(preds.numpy())
            P_resized=np.zeros((preds.shape[0],image.size[1],image.size[0]),dtype=float)
            for dim in range(preds.shape[0]):
                print("%s = %f"%(self.prompts[dim],P2[dim,0,:,:].max()))            
                P_resized[dim,:,:]=cv2.resize(P2[dim,0,:,:],(image.size[0],image.size[1]))
        else:
            preds=outputs.logits.unsqueeze(0)
            P2=self.sigmoid(preds.numpy())            
            P_resized=np.zeros((1,image.size[1],image.size[0]),dtype=float)
            P_resized[0]=cv2.resize(P2[0],(image.size[0],image.size[1]))

        return P_resized

class map_run():
    def __init__(self, prompts, color_image_dir:str, depth_image_dir:str, initialize_model=True):
        if initialize_model:
            self.CSmodel=clip_seg(prompts)
        else:
            self.CSmodel=None
        self.prompts=prompts
        self.egrid=evidence_grid3D([-1.0, -5.0, -0.3],[5.0, 0.5, 2.0],[60,55,23],len(prompts))
        self.K_rotated=[906.7647705078125, 0.0, 368.2167053222656, 
                        0.0, 906.78173828125, 650.24609375,                 
                        0.0, 0.0, 1.0]
        self.f_x=self.K_rotated[0]
        self.c_x=self.K_rotated[2]
        self.f_y=self.K_rotated[4]
        self.c_y=self.K_rotated[5]
        self.color_dir=color_image_dir
        self.depth_dir=depth_image_dir

    def get_3D_point(self, x_pixel, y_pixel, depth):
        x = (x_pixel - self.c_x) * depth / self.f_x
        y = (y_pixel - self.c_y) * depth / self.f_y
        return [x,y,depth]

    # def add_probability_grid(self, xyz_matrix, p_grid, num_samples=5000):
    #     if xyz_matrix.shape[1]!=p_grid.shape[1] or xyz_matrix[2]!=p_grid.shape[2]:
    #         print("XYZ Matrix and P-grid should have same 1+2 dimensions")
    #         return False
    #     if xyz_matrix.shape[0]!=3:
    #         print("XYZ matrix should have size 3 in first dimension")
    #     if p_grid.shape[0]!=self.num_inference_dim:
    #         print("P_grid inference dimensions do not match this evidence grid")
        
    #     # Strategy - randomly sample the image
    #     RS=np.floor(np.random.rand(num_samples,2)*p_grid.shape[1:]).astype(int)
    #     for idx in range(num_samples):
    #         self.add_evidence_logodds(xyz[:,RS[idx,0],RS[idx,1]],p_grid[:,RS[idx,0],RS[idx,1]])

    def add_image(self, color_fName, global_poseM, clipV, num_samples=5000, add_ray=False):
        # Get the depth image
        color_fName_s=color_fName.split('_')
        depth_fName=self.depth_dir+"/depth_"+color_fName_s[-1]
        depth_image=cv2.imread(depth_fName,cv2.IMREAD_UNCHANGED)
        if clipV[0].shape!=depth_image.shape:
            # Check for simple rotation
            if clipV.shape[1]==depth_image.shape[1] and clipV.shape[2]==depth_image.shape[0]:
                depth_image=cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
            else:
                raise Exception("Depth image shape does not match shape of color image")

        # Now randomly sample the image
        RS=np.floor(np.random.rand(num_samples,2)*depth_image.shape[:2]).astype(int)
        for idx in range(num_samples):
            dpt=depth_image[RS[idx,0],RS[idx,1]]
            if dpt>0 and dpt<4000:
                A=self.get_3D_point(RS[idx,1],RS[idx,0],dpt/1000.0)
                xyz=np.matmul(global_poseM,[A[0],A[1],A[2],1.0])[:3]
                if add_ray:
                    self.egrid.add_ray(xyz, global_poseM[:3,3], clipV[:,RS[idx,0],RS[idx,1]])
                else:
                    self.egrid.add_evidence_logodds(xyz, clipV[:,RS[idx,0],RS[idx,1]])
    
    def clip_inference(self, color_fName):
        if self.CSmodel is None:
            print("ClipSeg model not initialized")
            return None
        # Need to use PILLOW to load the color image - it has an impact on the clip model???
        image = Image.open(self.color_dir+"/"+color_fName)
        # Get the clip probabilities
        clipV=self.CSmodel.process_image(image)
        return clipV, image

    def save_inference(self, save_dir, color_fName, global_poseM, clipV):
        with open(save_dir+"/"+color_fName+".pkl","wb") as fout:
            pickle.dump(clipV, fout)
            pickle.dump(global_poseM, fout)
            pickle.dump(self.prompts, fout)
            pickle.dump(color_fName, fout)

    def load_inference(self, pickle_file):
        try:
            with open(pickle_file, "rb") as fin:
                clipV=pickle.load(fin)
                global_poseM=pickle.load(fin)
                prompts=pickle.load(fin)
                color_fName=pickle.load(fin)
                if prompts==self.prompts:
                    return color_fName, global_poseM, clipV
                else:
                    print("Prompts do not match. Indicated file contains the following prompts")
                    print(prompts)
                    return None
        except Exception as e:
            return None

    def process_and_add_image(self, color_fName, global_poseM, num_samples=5000, save_dir=None):
        try:
            clipV, image=self.clip_inference(color_fName)
        except Exception as e:
            print(e)
            return
        
        if save_dir is not None:
            self.save_inference(save_dir, color_fName, global_poseM, clipV)

        self.add_image(color_fName, global_poseM, clipV, num_samples)
        
    def generate_max_height_image(self):
        max_height_image=np.zeros((self.egrid.shape[0],self.egrid.shape[1],3),dtype=np.uint8)
        # colors = (np.random.rand(self.egrid.num_inference_dim,3)*255).astype(np.uint8)
        colors = [[0.8,0,0.1],
                  [0,1,0],
                  [0,0,1],
                  [1,0,1],
                  [1,1,0],
                  [0,1,1],
                  [0.4,0.6,0.3]]
        colors = (np.array(colors)*255).astype(np.uint8)
        gtzero=(self.egrid.grid>0).sum(0)
        whichD=self.egrid.grid.argmax(0)
        for cX in range(self.egrid.shape[0]):
            for cY in range(self.egrid.shape[1]):
                # This is ridiculous .. but I can't find a simple mask operator
                whichZ=np.where(gtzero[cX,cY,:])[0]
                if len(whichZ)>0:
                    max_height_image[cX,cY,:]=colors[whichD[cX,cY,whichZ[-1]],:]
        return max_height_image

from scipy.spatial.transform import Rotation as R
from image_set import image_set
import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('images_initial',type=str,help='location of initial pose csv file to process')
#     parser.add_argument('clip_initial',type=str,help='initial clip csv file to process')
#     parser.add_argument('images_change',type=str,help='location of change pose csv file to process')
#     parser.add_argument('clip_change',type=str,help='change clip csv file to process')
 
# GI=image_set('/home/emartinso/data/office/office_change/depthP_initial.csv')
# prompts=["a computer", "a suspicious object", "signs of a break-in", "a package", "wooden office furniture"]
# model=map_run(prompts, "/home/emartinso/data/office/office_change/initial/rotated", "/home/emartinso/data/office/office_change/initial/depth")
# names=[ im for im in GI.all_images.keys() ]
# np.random.shuffle(names)
# for idx,im in enumerate(names[:100]):
#     print(str(idx)+": " + im)
#     rotM=GI.all_images[im]['global_poseM']
#     model.process_and_add_image(im, rotM,save_fName=True)
# MI_image=model.generate_max_height_image()
# cv2.imshow("max height",MI_image)
# cv2.waitKey(0)
# pdb.set_trace()

# fName='rgb_00522.png'
# rotM=np.identity(4)
# rotM[:3,3]=[1.847, -0.529, 1.315]
# quat=[0.341398, 0.729780, -0.544791, -0.232533]
# rotM[:3,:3]=R.from_quat(quat).as_matrix()
# # fName="rgb_00175.png"
# MI_image=model.generate_max_height_image()
# cv2.imshow("max height",MI_image)
# cv2.waitKey(0)
# pdb.set_trace()
