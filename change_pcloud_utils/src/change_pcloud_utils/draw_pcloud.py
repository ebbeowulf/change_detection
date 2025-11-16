# Utility class for drawing point clouds onto a 2D image
#    Pointclouds are in the open3d format

import numpy as np
import copy
import cv2
import pdb

MAX_POINTS=200000

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

        # Have to clear NaN pixels - open3d doesn't do it for us
        validP=np.where(np.isnan(self.xyz).sum(1)==0)
        self.xyz=self.xyz[validP]
        self.rgb=self.rgb[validP]

        self.minXYZ=self.xyz.min(0)-0.5
        self.maxXYZ=self.xyz.max(0)+0.5
        self.dy_dr=(self.maxXYZ-self.minXYZ)[1]/self.height
        self.dx_dc=(self.maxXYZ-self.minXYZ)[0]/self.width

        self.window_name=window_name
        self.bg_image=self.draw_all_dots(height_range=[0.1,2.0])
        self.fg_image=self.bg_image

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
                    clr=(tmpC[2],tmpC[1],tmpC[0])
                cv2.circle(image, (col, row), radius=2,color=clr,thickness=-1)
            except Exception as e:
                print("Exception: " + str(e))
                print("skipping pixel during dot printing")
        return image

    def draw_all_dots(self, height_range=[-1.0, 100], bg_image=None):
        whichP=np.where((self.xyz[:,2]<height_range[1])*(self.xyz[:,2]>height_range[0]))
        return self.draw_dots(self.xyz[whichP], self.rgb[whichP], bg_image=bg_image)

    def xyz_to_rc(self, xyz_pt):
        row=self.height-int((xyz_pt[1]-self.minXYZ[1])/self.dy_dr)
        col=int((xyz_pt[0]-self.minXYZ[0])/self.dx_dc)
        return row,col

    def rc_to_xy(self, row, col):
        y=(self.height-row)*self.dy_dr+self.minXYZ[1]
        # row=self.height-int((xyz_pt[1]-self.minXYZ[1])/self.dy_dr)
        x=col*self.dx_dc+self.minXYZ[0]
        # col=int((xyz_pt[0]-self.minXYZ[0])/self.dx_dc)
        return x,y
        
    def overlay_dots(self, xyz:np.array, fixed_color=(0,0,255)):
        return self.draw_dots(xyz,local_rgb=fixed_color, bg_image=self.bg_image)
    
    def add_boxes_to_fg(self, boxes, fixed_color=(0,0,255)):
        image=copy.copy(self.fg_image)
        for box in boxes:
            if len(box)>3: #shape = (6,)
                row1,col1=self.xyz_to_rc(box[:3])
                row2,col2=self.xyz_to_rc(box[3:])
            else: #shape = (2,3)
                row1,col1=self.xyz_to_rc(box[0])
                row2,col2=self.xyz_to_rc(box[1])
            image=cv2.rectangle(image, (col1,row1), (col2,row2), color=fixed_color)
        self.set_fg_image(image)

    def set_fg_image(self, cv_image:np.array):
        self.fg_image=copy.copy(cv_image)

    def draw_fg(self):
        cv2.imshow(self.window_name, self.fg_image)
        return cv2.waitKey(0)

    def save_fg(self, fName):
        cv2.imwrite(fName, self.fg_image)