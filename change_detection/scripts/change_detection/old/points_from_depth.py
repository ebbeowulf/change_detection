import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import pdb

def get_world_pose(trans, quat):
    mm=np.identity(4)
    mm[:3,:3]=R.from_quat(quat).as_matrix()
    # mm=tf.transformations.quaternion_matrix(quat)
    mmT=np.transpose(mm)
    pose=np.matmul(-mmT[:3,:3],trans)
    mmT[:3,3]=pose
    return pose, mmT
    
def read_image_csv(images_txt):
    with open(images_txt,"r") as fin:
        A=fin.readlines()

    all_images={}
    for ln_ in A:
        if len(ln_)<2:
            continue
        if ln_[-1]=='\n':
            ln_=ln_[:-1]
        if ln_[0]=='#':
            continue
        if ',' in ln_:
            ln_s=ln_.split(', ')
        else:
            ln_s=ln_.split(' ')

        if len(ln_s)==10:
            # We will assume a format of {rootname}_{image_id}.png in the image name
            id_str=ln_s[-1].split('_')[-1].split('.')[0]
            id=int(id_str)
            quat=[float(ln_s[2]),float(ln_s[3]), float(ln_s[4]), float(ln_s[1])]
            trans=[float(ln_s[5]),float(ln_s[6]),float(ln_s[7])]
            world_pose, rotM = get_world_pose(trans, quat)
            image={'rot': quat, 'trans': trans, 'id': id, 'global_pose': world_pose, 'global_poseM': rotM, 'name': ln_s[-1]}
            all_images[ln_s[-1]]=image
    return all_images

class points_from_depth():
    def __init__(self, image_txt:str, depth_img_dir:str, K=None):
        if K is None:
            # apply defaults for rotated stretch camera image
            self.K=[906.78173828125, 0.0, 650.24609375, 0.0, 906.7647705078125, 368.2167053222656, 0.0, 0.0, 1.0]
        else:
            self.K=K
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
        fName=self.image_directory+"/depth_%05d.png"%(self.all_images[im_name]['id'])
        try:
            depth_img=cv2.imread(fName,cv2.IMREAD_UNCHANGED)
            return depth_img.astype(float)/1000.0
        except Exception as e:
            print("Failed to open depth image")
            return None

    def get_center_point(self, im_name):
        depthI=self.get_depth_image(im_name)
        if depthI is None:
            return None
        half_s=np.round(np.array(depthI.shape)/2.0).astype('int')
        depths = []
        for row in range(half_s[0]-3,half_s[0]+3):
            for col in range(half_s[1]-3,half_s[1]+3):
                if depthI[row,col]>0 and depthI[row,col]<4000:
                    depths.append(depthI[row,col])
        if len(depths)<1:
            return None
        center_depth=np.median(depths)
        xyz=np.array([0,0,0,1])
        xyz[:3]=self.get_3D_point(half_s[1],half_s[0],center_depth)
        return np.matmul(self.all_images[im_name]['global_poseM'],xyz)
    
    def write_points_file(self, fName_out:str):
        with open(fName_out,'w') as fout:
            print("image, PX, PY, PZ, QX, QY, QZ, QW, CenterPX, CenterPY, CenterPZ",file=fout)
            for key in self.all_images.keys():
                ctrP=self.get_center_point(key)
                if ctrP is None:
                    continue
                im=self.all_images[key]
                quat=R.from_matrix(im['global_poseM'][:3,:3]).as_quat()

                print(key,file=fout,end=', ')
                print("%0.3f, %0.3f, %0.3f"%(im['global_pose'][0], im['global_pose'][1], im['global_pose'][2]),file=fout,end=', ')
                print("%f, %f, %f, %f"%(quat[0], quat[1], quat[2], quat[3]),file=fout,end=', ')
                print("%0.3f, %0.3f, %0.3f"%(ctrP[0], ctrP[1], ctrP[2]), file=fout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_txt',type=str,help='location of initial pose file generated by colmap to process')
    parser.add_argument('depth_image_dir',type=str,help='location of the depth images')
    parser.add_argument('out_file',type=str,help='location to write the results')
    args = parser.parse_args()

    PD=points_from_depth(args.image_txt, args.depth_image_dir)
    PD.write_points_file(args.out_file)





