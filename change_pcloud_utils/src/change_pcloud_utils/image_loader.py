import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2
DEVICE='cuda'
from change_pcloud_utils.map_utils import get_rotated_points

class image_loader():
    def __init__(self, params):
        self.params=params
        self.rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy,device=DEVICE)
        self.cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx,device=DEVICE)
        self.rot_matrixT=torch.tensor(params.rot_matrix,device=DEVICE)        
        self.loaded_image=None

    def load_image(self, 
                   colorI_fName,
                   depthI_fName,
                   poseM:np.ndarray, 
                   max_distance=10.0):
        try:
            if self.loaded_image is None:
                self.loaded_image=dict()
            self.loaded_image['color']=Image.open(colorI_fName)
            depthI_np=cv2.imread(depthI_fName,-1)
            self.loaded_image['depthT']=torch.tensor(depthI_np.astype('float')/1000.0,device=DEVICE)
            self.loaded_image['colorT']=torch.tensor(np.array(self.loaded_image['color']),device=DEVICE)
            self.loaded_image['x'] = self.cols*self.loaded_image['depthT']/self.params.fx
            self.loaded_image['y'] = self.rows*self.loaded_image['depthT']/self.params.fy
            self.loaded_image['depth_mask']=(self.loaded_image['depthT']>1e-4)*(self.loaded_image['depthT']<max_distance)
            self.loaded_image['nonzero_rows'], self.loaded_image['nonzero_cols'] = self.loaded_image['depth_mask'].nonzero(as_tuple=True)

            # Build the rotation matrix
            self.loaded_image['M']=torch.matmul(self.rot_matrixT,torch.tensor(poseM,device=DEVICE))

            self.loaded_image['pts']=get_rotated_points(self.loaded_image['x'],
                                                        self.loaded_image['y'],
                                                        self.loaded_image['depthT'],
                                                        self.loaded_image['depth_mask'],
                                                        self.loaded_image['M'])
            return True
        except Exception as e:
            print(f"Failed to load image materials for {colorI_fName}")
            self.loaded_image=None
        return False

    def calculate_dist_to_closest_point(self, xyz):
        if type(xyz)==torch.tensor:
            xyzT=xyz
        else:
            xyzT=torch.tensor(xyz,device=DEVICE)
        deltaP=self.loaded_image['pts']-xyzT
        return ((deltaP**2).sum(1)).min().sqrt().cpu().tolist()
        
    def get_closest_row_col(self, xyz):
        # same as calc dist to closest point, but need to get the argmin
        #   and pull from nonzero row/col
        if type(xyz)==torch.tensor:
            xyzT=xyz
        else:
            xyzT=torch.tensor(xyz,device=DEVICE)
        deltaP=self.loaded_image['pts']-xyzT
        whichP=torch.argmin((deltaP**2).sum(1))
        return self.loaded_image['nonzero_rows'][whichP].cpu().tolist(),self.loaded_image['nonzero_cols'][whichP].cpu().tolist()
                    