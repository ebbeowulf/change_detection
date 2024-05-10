from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
import rospy
import numpy as np
import pdb
from cv_bridge import CvBridge
import cv2
import copy
import glob

ROOT_DIR="/home/ebeowulf/projects/ScanNet/data/scans/scene0706_00/raw_output/"
CAMINFO_FILE="/home/ebeowulf/projects/ScanNet/data/scans/scene0706_00/scene0706_00.txt"

def load_camera_info(info_file):
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            if key=='sceneType':
                if val[-1]=='\n':
                    val=val[:-1]
                info_dict[key] = val
            elif key=='axisAlignment':
                info_dict[key] = np.fromstring(val, sep=' ')
            else:
                info_dict[key] = float(val)

    if 'axisAlignment' not in info_dict:
       info_dict['rot_matrix'] = np.identity(4)
    else:
        info_dict['rot_matrix'] = info_dict['axisAlignment'].reshape(4, 4)
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

        rospy.Timer(rospy.Duration(2.0),self.timer_cb)
        self.last_counter=-1
        self.all_files = glob.glob(root_dir+"/*.txt")
        self.counter=0
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
            import sys
            sys.exit(-1)

        if self.all_files[self.counter].endswith(".pose.txt"):            
            if not (self.publish_single(self.get_current_root_file())):
                self.counter+=1
        else:
            self.counter+=1

    def get_current_root_file(self):
        return self.all_files[self.counter][0:-9]
    
    def publish_single(self, file_root):
        print("Publish: " + file_root)
        H=Header()
        H.frame_id="camera"
        H.stamp=rospy.Time.now()
        try:
            colorI=cv2.imread(file_root+".color.jpg",-1)
            depthI=cv2.imread(file_root+".depth.pgm",-1)     

            cinfo_msg=copy.copy(self.cinfo_color)
            cinfo_msg.header=H
            self.pub_colorInfo.publish(cinfo_msg)
            colorMsg=self.br.cv2_to_imgmsg(colorI,"bgr8")
            colorMsg.header=H
            self.pub_color.publish(colorMsg)

            cinfo_msg=copy.copy(self.cinfo_depth)
            cinfo_msg.header=H
            self.pub_depthInfo.publish(cinfo_msg)
            depthMsg=self.br.cv2_to_imgmsg(depthI)
            depthMsg.header=H
            self.pub_depth.publish(depthMsg)
            return True
        except Exception as e:
            print("Failed to publish")
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
    RP=publish_and_register(ROOT_DIR, CAMINFO_FILE)
    rospy.spin()