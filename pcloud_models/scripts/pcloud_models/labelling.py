import open3d as o3d
import numpy as np
from map_utils import rgbd_file_list
import argparse
import cv2
import pickle
from map_from_scannet import build_file_structure
import pdb
import time
import os
import copy

MAX_POINTS=200000

# Global Variables
mouse_origin=None
image_interface=None

def mouse_cb(event,x,y,flags,param):
    global mouse_origin, image_interface
    print("Mouse event: " + str(event))
    if event == cv2.EVENT_LBUTTONDOWN:
        print("left button down")
        mouse_origin=(x,y)
    elif event == cv2.EVENT_MOUSEMOVE and mouse_origin is not None:
        print("draw arrow")
    elif event == cv2.EVENT_LBUTTONUP and mouse_origin is not None:
        print("open images")
        mouse_origin=None

class drawn_image():
    def __init__(self, pointcloud, width=1280, height=720, window_name="environment"):
        self.image=np.zeros((height,width,3),dtype=np.uint8)
        self.height=height
        self.width=width

        pct=MAX_POINTS / len(pointcloud.points)
        if pct<1.0:
            sampled=pointcloud.random_down_sample(pct)
            self.xyz=np.array(sampled.points)
            self.rgb=np.array(sampled.colors)
        else:
            self.xyz=np.array(pointcloud.points)
            self.rgb=np.array(pointcloud.colors)

        self.minXYZ=self.xyz.min(0)-0.5
        self.maxXYZ=self.xyz.max(0)+0.5
        self.dy_dr=(self.maxXYZ-self.minXYZ)[1]/self.height
        self.dx_dc=(self.maxXYZ-self.minXYZ)[0]/self.width

        self.window_name=window_name
        self.bg_image=self.draw_all_dots(height_range=[0.1,2.0])

    # Draw the dots with the specified color, returning the image
    #   optionally start with a background image. local_rgb should either be an array of size (N,3)
    #   where N is the same size as local_xyz, or else a tuple of 3 values
    def draw_dots(self, local_xyz:np.array, local_rgb, bg_image=None):
        if bg_image is not None:
            image=copy.copy(bg_image)
        else:
            image=255*np.ones((self.height,self.width,3),dtype=np.uint8)

        rr=np.argsort(local_xyz[:,2])

        if type(local_rgb)==tuple:
            clr=local_rgb
            dynamic_color=False
        elif local_rgb.shape[0]==local_xyz.shape[0]:
            dynamic_color=True
        else:
            raise Exception("Color array is incorrect - not drawing")

        for idx in range(len(rr)):
            try:
                row,col=self.xyz_to_rc(local_xyz[rr[idx]])
                if dynamic_color:
                    tmpC=(local_rgb[rr[idx]]*255).astype(int).tolist()
                    clr=(tmpC[0],tmpC[1],tmpC[2])
                cv2.circle(image, (col, row), radius=2,color=clr,thickness=-1)
            except Exception as e:
                print("Exception: " + str(e))
                pdb.set_trace()
        return image

    def draw_all_dots(self, height_range=[-1.0, 100], bg_image=None):
        whichP=np.where((self.xyz[:,2]<height_range[1])*(self.xyz[:,2]>height_range[0]))
        return self.draw_dots(self.xyz[whichP], self.rgb[whichP], bg_image=bg_image)

    def xyz_to_rc(self, xyz_pt):
        row=self.height-int((xyz_pt[1]-self.minXYZ[1])/self.dy_dr)
        col=int((xyz_pt[0]-self.minXYZ[0])/self.dx_dc)
        return row,col
    
    def overlay_dots_and_box(self, xyz:np.array, fixed_color=(0,0,255)):
        image=self.draw_dots(xyz,local_rgb=fixed_color, bg_image=self.bg_image)
        minP=xyz.min(0)
        maxP=xyz.max(0)
        minR,minC=self.xyz_to_rc(minP)
        maxR,maxC=self.xyz_to_rc(maxP)
        return cv2.rectangle(image,(minC,minR),(maxC,maxR),color=fixed_color,thickness=2)

def label_scannet_pcloud(root_dir, raw_dir, save_dir, targets):
    global image_interface

    fList=build_file_structure(root_dir+"/"+raw_dir, root_dir+"/"+save_dir)

    # Pull the point cloud for drawing a background
    s2=root_dir
    if s2[-1]=="/":
        s2=s2[:-1]
    
    window_name=root_dir
    # cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow(window_name)
    # pcloud_model_fName=root_dir+"/"+s2.split('/')[-1]+"_vh_clean_2.ply"
    pcl_raw=o3d.io.read_point_cloud(fList.get_combined_pcloud_fileName())
    image_interface=drawn_image(pcl_raw,window_name=window_name)
    cv2.setMouseCallback(window_name, mouse_cb)

    #Load target pcloud
    for tgt in targets:
        print(tgt)
        ply_fileName=fList.get_combined_pcloud_fileName(tgt)
        try:
            pcl_target=o3d.io.read_point_cloud(ply_fileName)
            if pcl_target is None:
                raise Exception("file not found")
        except Exception as e:
            print(ply_fileName + " - not found")
            continue

        tgt_pts=np.array(pcl_target.points)
        image=image_interface.overlay_dots_and_box(tgt_pts)
        while(1):
            cv2.imshow(window_name, image)
            # cv2.imshow(window_name, image_interface.bg_image)
            input_key=cv2.waitKey(0)
            if input_key==ord('n'):
                print("moving to next object")
                break
            elif input_key==ord('q'):
                print("exiting labeling task")
                import sys
                sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    parser.set_defaults(yolo=True)
    # parser.add_argument('--targets',type=list, nargs='+', default=None, help='Set of target classes to build point clouds for')
    args = parser.parse_args()
    label_scannet_pcloud(args.root_dir, args.raw_dir, args.save_dir, args.targets)
