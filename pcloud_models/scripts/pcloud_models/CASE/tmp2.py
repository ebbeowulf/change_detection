import numpy as np
import cv2
import json
from camera_params import camera_params
import torch
import pdb
import matplotlib.pyplot as plt

def get_camera_params(json_file):
    with open(json_file,'r') as infile:
        A=json.load(infile)
    return camera_params(A['h'],A['w'],A['fl_x'],A['fl_y'],A['cx'],A['cy'],np.identity(4))

def get_rot_matrix(json_file, image_uid):
    with open(json_file,'r') as infile:
        A=json.load(infile)
    for frame in A['frames']:
        if image_uid in frame['file_path']:
             tf=np.array(frame['transform_matrix'])
             return torch.tensor(tf)
    
    return None

root_dir="/home/emartinso/data/living_room/DATA_DIR_clean2_4/hat_1/"
json_file=root_dir + "transforms.json"
img_uid='0098'
rgb_file=root_dir + f"renders/rgb_{img_uid}.png"
depth_file=root_dir + f"renders/depth_{img_uid}.png"

params=get_camera_params(json_file)    
rot_matrixT=get_rot_matrix(json_file, img_uid)

rows=torch.tensor(np.tile(np.arange(params.height).reshape(params.height,1),(1,params.width))-params.cy)
cols=torch.tensor(np.tile(np.arange(params.width),(params.height,1))-params.cx)

colorI=cv2.imread(rgb_file, -1)
depthI_orig=cv2.imread(depth_file, -1)
kernel = np.ones((11,11),np.float32)/121
depthI = cv2.filter2D(depthI_orig,-1,kernel)
# plt.imshow(depthI)
# plt.show()

depthT=torch.tensor(depthI.astype('float')/1000.0)
colorT=torch.tensor(colorI)
x = cols*depthT/params.fx
y = rows*depthT/params.fy

depth_mask=(depthT>1e-4)*(depthT<1.0)
pts=torch.stack([x[depth_mask],y[depth_mask],depthT[depth_mask],torch.ones(((depth_mask>0).sum()))],dim=1)
pts_rot=torch.matmul(rot_matrixT,pts.transpose(0,1))
pts_rot=pts_rot[:3,:].transpose(0,1)

# p1=pts_rot.cpu().numpy()
p1=pts[:,:3].cpu().numpy()
c1=colorT[depth_mask].cpu().numpy()
pdb.set_trace()

import open3d as o3d
from map_utils import pointcloud_open3d

pcd=pointcloud_open3d(p1,c1)
o3d.visualization.draw_geometries([pcd])
pdb.set_trace()
