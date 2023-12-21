import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
from image_set import read_image_csv

class points_from_depth():
    def __init__(self, image_txt:str, depth_img_dir:str):
        self.K=[906.78173828125, 0.0, 650.24609375, 0.0, 906.7647705078125, 368.2167053222656, 0.0, 0.0, 1.0]
        self.f_x=self.K[0]
        self.c_x=self.K[2]
        self.f_y=self.K[4]
        self.c_y=self.K[5]
        self.image_directory=depth_img_dir
        self.all_images=read_image_csv(image_txt)

    def get_3D_point(self, x_pixel, y_pixel, depth):
        x = (x_pixel - self.c_x) * depth / self.f_x
        y = (y_pixel - self.c_y) * depth / self.f_y
        return [x,y,depth]

    def get_depth_image(self, im_name):
        fName=self.image_directory+"/depth_%5d.png"%(self.all_images[im_name]['id'])
        try:
            depth_img=cv2.open(fName)
            return depth_img
        except Exception as e:
            print("Failed to open depth image")
            return None

    def get_center_point(self, im_name):
        half_s=np.round(im_name.shape/2.0)
        depths = []
        for a in range(half_s[0]-3,half_s[0]+3):
            for b in range(half_s[1]-3,half_s[1]+3):
                if im_name[a,b]>0 and im_name[a,b]<8000:
                    depths.append(im_name[a,b])
        if len(depths)<1:
            return None
        center_depth=np.median(depths)
        return self.get_3D_point(half_s[0],half_s[1],center_depth)
    
    def write_points_file(self, fName_out:str):
        with open(fName_out,'w') as fout:
            print("image, PX, PY, PZ, QX, QY, QZ, QW, CenterPX, CenterPY, CenterPZ",file=fout)
            for key in self.all_images.keys():
                ctrP=self.get_center_point(key)
                if ctrP is None:
                    continue
                im=self.all_images[key]
                quat=R.from_matrix(im['global_poseM'][:3,:3]).as_quat()

                print("%s, %0.3f, %0.3f, %0.3f, %f, %f, %f, %0.3f, %0.3f, %0.3f"%(key,
                                                                                  im['global_pose'][0], im['global_pose'][1], im['global_pose'][2], 
                                                                                  quat[0], quat[1], quat[2], quat[3],
                                                                                  ctrP[0], ctrP[1], ctrP[2]), file=fout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_txt',type=str,help='location of initial pose file to process')
    parser.add_argument('depth_image_dir',type=str,help='location of the depth images')
    parser.add_argument('out_file',type=str,help='location to write the results')
    args = parser.parse_args()

    PD=points_from_depth(args.image_txt)
    PD.write_points_file(args.out_file)





