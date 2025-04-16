#!/usr/bin/env python3

from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
import rospy
import numpy as np
import pdb
from cv_bridge import CvBridge
import cv2
import copy
import glob
import tf
from geometry_msgs.msg import TransformStamped
import sys
import tf2_ros
import argparse

ROOT_DIR="/home/hindurthi/thesis_data/python_files/scannet/scans/scene0050_00/raw_output/"
CAMINFO_FILE="/home/hindurthi/thesis_data/python_files/scannet/scans/scene0050_00/scene0050_00.txt"

# ROOT_DIR="/home/ebeowulf/projects/ScanNet/data/scans/scene0706_00/raw_output/"
# CAMINFO_FILE="/home/ebeowulf/projects/ScanNet/data/scans/scene0706_00/scene0706_00.txt"

def static_transform_broadcaster(parent_frame, child_frame, transform_matrix):
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = parent_frame
    static_transformStamped.child_frame_id = child_frame

    static_transformStamped.transform.translation.x = transform_matrix[0,3]
    static_transformStamped.transform.translation.y = transform_matrix[1,3]
    static_transformStamped.transform.translation.z = transform_matrix[2,3]

    quat = tf.transformations.quaternion_from_matrix(transform_matrix)
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]

    broadcaster.sendTransform(static_transformStamped)

def load_camera_info(info_file):
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            if line[-1]=='\n':
                line=line[:-1]
            (key, val) = line.split(" = ")
            if key=='sceneType':
                info_dict[key] = val
            elif key=='axisAlignment' or key=='colorToDepthExtrinsics':
                info_dict[key] = np.fromstring(val, sep=' ')
            elif key=='colorToDepthExtrinsics':
                info_dict[key] = np.fromstring(val, sep=' ')
            else:
                try:
                    info_dict[key] = float(val)
                except Exception as e:
                    info_dict[key] = val

    return info_dict

class publish_and_register():
    def __init__(self, root_dir, caminfo_file):
        self.create_camera_info(caminfo_file)
        self.root_dir=root_dir
        self.pub_color=rospy.Publisher("/camera/rgb/image_rect_color",Image,queue_size=1)
        self.pub_depth=rospy.Publisher("/camera/depth/image_rect_raw",Image,queue_size=1)
        self.pub_colorInfo=rospy.Publisher("/camera/rgb/camera_info",CameraInfo,queue_size=1)
        self.pub_depthInfo=rospy.Publisher("/camera/depth/camera_info",CameraInfo,queue_size=1)
        self.pub_trigger=rospy.Subscriber("/camera/depth_registered/image_rect",Image,self.depth_reg_sub)

        self.br = CvBridge()

        self.counter=0
        rospy.Timer(rospy.Duration(2.0),self.timer_cb)
        self.last_counter=-1
        self.all_files = glob.glob(root_dir+"/*.txt")
        rospy.sleep(1.0)
        self.next()
    
    def timer_cb(self, tmr):
        # Need a way of skipping if something has frozen the system
        if self.last_counter==self.counter:
            self.counter+=1
            self.next()
        self.last_counter=self.counter
        
    def next(self):
        if self.counter>=len(self.all_files):
            print("Finished")
            rospy.signal_shutdown("Finished ... quitting now")

        if self.all_files[self.counter].endswith(".pose.txt"):            
            if not (self.publish_single(self.get_current_root_file())):
                self.counter+=1
        else:
            self.counter+=1

    # def next(self):
    #     self.publish_single(self.all_files[0][0:-9])

    def get_current_root_file(self):
        return self.all_files[self.counter][0:-9]
    
    def publish_single(self, file_root):
        print("Publish: " + file_root)
        tstamp=rospy.Time.now()
        try:
            colorI=cv2.imread(file_root+".color.jpg",-1)
            depthI=cv2.imread(file_root+".depth.pgm",-1)     

            # PUblish the color camera info message
            cinfo_msg=copy.copy(self.cinfo_color)
            cinfo_msg.header.stamp=tstamp
            cinfo_msg.header.frame_id=self.color_frame
            self.pub_colorInfo.publish(cinfo_msg)

            # Publish the color image
            colorMsg=self.br.cv2_to_imgmsg(colorI,"bgr8")
            colorMsg.header.stamp=tstamp
            colorMsg.header.frame_id=self.color_frame
            self.pub_color.publish(colorMsg)

            # Publish the depth camera info message
            dinfo_msg=copy.copy(self.cinfo_depth)
            dinfo_msg.header.stamp=tstamp
            dinfo_msg.header.frame_id=self.depth_frame
            self.pub_depthInfo.publish(dinfo_msg)

            # Publish the depth image
            depthMsg=self.br.cv2_to_imgmsg(depthI)
            depthMsg.header.stamp=tstamp
            depthMsg.header.frame_id=self.depth_frame
            self.pub_depth.publish(depthMsg)
            return True
        except Exception as e:
<<<<<<< HEAD
            print("Failed to publish")
            print(e)
=======
            print(f"Failed to publish {e}")
>>>>>>> 773aa37985836ddc92484d34a04f5972cd9cf75e
        return False

    def create_camera_info(self, caminfo_file):
        # Rotating the mesh to axis aligned
        self.info_dict=load_camera_info(caminfo_file)
        
        self.cinfo_color=CameraInfo()
        self.cinfo_color.distortion_model="pinhole"
        self.cinfo_color.header.frame_id="camera"
        self.cinfo_color.height=int(self.info_dict['colorHeight'])
        self.cinfo_color.width=int(self.info_dict['colorWidth'])
        self.cinfo_color.D=[0,0,0,0,0]
        self.cinfo_color.K=[self.info_dict['fx_color'], 0, self.info_dict['mx_color'], 0, self.info_dict['fy_color'], self.info_dict['my_color'], 0, 0, 1]
        self.cinfo_color.P=[self.info_dict['fx_color'], 0, self.info_dict['mx_color'], 0, 
                            0, self.info_dict['fy_color'], self.info_dict['my_color'], 0, 
                            0, 0, 1, 0]

        self.cinfo_depth=CameraInfo()
        self.cinfo_depth.distortion_model="pinhole"
        self.cinfo_depth.header.frame_id="camera"
        self.cinfo_depth.height=int(self.info_dict['depthHeight'])
        self.cinfo_depth.width=int(self.info_dict['depthWidth'])
        self.cinfo_depth.D=[0,0,0,0,0]
        self.cinfo_depth.K=[self.info_dict['fx_depth'], 0, self.info_dict['mx_depth'], 0, self.info_dict['fy_depth'], self.info_dict['my_depth'], 0, 0, 1]
        self.cinfo_depth.P=[self.info_dict['fx_depth'], 0, self.info_dict['mx_depth'], 0, 
                            0, self.info_dict['fy_depth'], self.info_dict['my_depth'], 0, 
                            0, 0, 1, 0]

        # Even though a transform was provided - it looks like the depth
        #   images in ScanNet have already been aligned ... adding the extrinsics
        #   worsens the alignment, at least in some cases
        if 1: #'colorToDepthExtrinsics' not in self.info_dict:
            self.depth_frame="camera/depth"
            self.color_frame="camera/rgb"
            extrinsics=np.identity(4)
        else:
            extrinsics = self.info_dict['colorToDepthExtrinsics'].reshape(4, 4)
            self.depth_frame="camera/depth"
            self.color_frame="camera/rgb"
        static_transform_broadcaster(self.color_frame, self.depth_frame, extrinsics)



    def depth_reg_sub(self, image_msg:Image):
        print("Image received")
        depth_registered=self.br.imgmsg_to_cv2(image_msg)
        fName_out=self.get_current_root_file()+".depth_reg.png"
        cv2.imwrite(fName_out, depth_registered)
        self.counter+=1
        self.next()


# def load_and_rectify(self, file_root, cinfo_color, cinfo_depth):
#     colorI=cv2.imread(file_root+".color.jpg",-1)
#     depthI=cv2.imread(file_root+".depth.pgm",-1)

    
    
if __name__ == '__main__':
    rospy.init_node("scannet_publisher")
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',type=str,help='location of scannet directory to process')
    args = parser.parse_args()
    r_split=args.root_dir.split('/')
    if r_split[-1]=='':
        cam_info=args.root_dir+r_split[-2]+".txt"
    else:
        cam_info=args.root_dir+"/" + r_split[-1]+".txt"
    RP=publish_and_register(args.root_dir+"/raw_output/", cam_info)
    rospy.spin()