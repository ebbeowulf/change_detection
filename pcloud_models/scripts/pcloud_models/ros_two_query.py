#!/usr/bin/env python3
import rospy
import cv2
import pdb
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
import numpy as np
from threading import Lock
import tf
import message_filters
from camera_params import camera_params
from map_utils import pcloud_from_images, pointcloud_open3d
from std_srvs.srv import Trigger, TriggerResponse
from two_query_localize import create_object_clusters, calculate_iou, estimate_likelihood

TRACK_COLOR=True

def normalizeAngle(angle):
    while angle>=np.pi:
        angle-=2*np.pi
    while angle<-np.pi:
        angle+=2*np.pi
    return angle

class two_query_localize:
    def __init__(self, main_query, llm_query, cluster_min_points, detection_threshold, travel_params, storage_dir=None):
        # Initization of the node, name_sub
        rospy.init_node('two_query_localize', anonymous=True)
        self.listener = tf.TransformListener()
        self.storage_dir = storage_dir

        # Initialize the CvBridge class
        self.bridge = CvBridge()
        self.im_count=1

        self.pose_queue=[]
        self.pose_lock=Lock()
        self.pose_sub = rospy.Subscriber("/odom", Odometry, self.pose_callback)
        
        # Last image stats
        self.last_image=None
        self.travel_params=travel_params

        # Create Pcloud Data Structures
        self.pcloud_creator=None
        self.query_main=main_query
        self.query_llm=llm_query
        self.pcloud_main={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
        self.pcloud_llm={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
        self.cluster_min_points=cluster_min_points
        self.detection_threshold=detection_threshold

        # Setup callback function
        self.camera_params_sub = rospy.Subscriber('/camera_throttled/depth/camera_info', CameraInfo, self.cam_info_callback)
        self.rgb_sub = message_filters.Subscriber('/camera_throttled/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera_throttled/depth/image_rect_raw', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.rgbd_callback)

        # Setup service calls
        self.clear_srv = rospy.Service('clear_clouds', Trigger, self.clear_clouds_service)
        self.print_srv = rospy.Service('print_clouds', Trigger, self.print_clouds_service)
        self.draw_clusters_srv = rospy.Service('draw_clusters', Trigger, self.draw_clusters_service)

    def clear_clouds_service(self, msg):
        resp=TriggerResponse()
        resp.success=True
        self.pcloud_main={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
        self.pcloud_llm={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
        self.last_image=None
        resp.message="clouds cleared"
        return resp

    def print_clouds_service(self, msg):
        resp=TriggerResponse()
        resp.success=False
        if self.pcloud_creator is None:
            resp.message="Pcloud creator not initialized"
            return resp
        import open3d as o3d
        if TRACK_COLOR:
            pcd_main=pointcloud_open3d(self.pcloud_main['xyz'],self.pcloud_main['rgb'])
            pcd_llm=pointcloud_open3d(self.pcloud_llm['xyz'],self.pcloud_llm['rgb'])
        else:
            resp.message="Color from probability ... not implemented yet"
            return resp
        fileName_main=f"{self.query_main}.{self.pcloud_main['xyz'].shape[0]}.ply"
        fileName_llm=f"{self.query_llm}.{self.pcloud_llm['xyz'].shape[0]}.ply"
        if self.storage_dir:
            fileName_llm=self.storage_dir+"/"+fileName_llm
            fileName_main=self.storage_dir+"/"+fileName_main
        o3d.io.write_point_cloud(fileName_main,pcd_main)
        o3d.io.write_point_cloud(fileName_llm,pcd_llm)
        resp.success=True
        resp.message=f"{fileName_main},{fileName_llm}"
        return resp

    # create the object clusters and filter by the number of points in each
    def get_clusters_ros(self, pcd):
        all_objects=create_object_clusters(pcd['xyz'],pcd['probs'], -1.0, self.detection_threshold)
        objects_out=[]
        for obj in all_objects:
            if obj.size()>self.cluster_min_points:
                objects_out.append(obj)
        return objects_out

    # create the clusters and match them, returning any that exceed the detection threshold
    def match_clusters(self, method):
        objects_main=self.get_clusters_ros(self.pcloud_main)
        objects_llm=self.get_clusters_ros(self.pcloud_llm)
        print(f"Clustering.... main {len(objects_main)}, llm {len(objects_llm)}")
        pdb.set_trace()
        positive_clusters=[]
        # Match with other clusters
        if method=='main':
            return objects_main
        elif method=='llm':
            return objects_llm
        else:
            for idx0 in range(len(objects_main)):                    
                cl_stats=[idx0, objects_main[idx0].prob_stats['max'], objects_main[idx0].prob_stats['mean'], -1, -1]
                for idx1 in range(len(objects_llm)):
                    IOU=calculate_iou(objects_main[idx0].box[0],objects_main[idx0].box[1],objects_llm[idx1].box[0],objects_llm[idx1].box[1])
                    if IOU>0.5:
                        cl_stats[3]=max(cl_stats[3],objects_llm[idx1].prob_stats['max'])
                        cl_stats[4]=max(cl_stats[4],objects_llm[idx1].prob_stats['mean'])

                if method=='llm_mean' and estimate_likelihood(cl_stats, 2)>self.detection_threshold:
                    positive_clusters.append(objects_main[idx0])
                elif method=='llm_max' and estimate_likelihood(cl_stats, 3)>self.detection_threshold:
                    positive_clusters.append(objects_main[idx0])
                elif method=='combo_max' and estimate_likelihood(cl_stats, 4)>self.detection_threshold:
                    positive_clusters.append(objects_main[idx0])
                elif method=='combo_mean' and estimate_likelihood(cl_stats, 5)>self.detection_threshold:
                    positive_clusters.append(objects_main[idx0])
        return positive_clusters
    
    def draw_clusters_service(self, msg):
        positive_clusters=self.match_clusters('combo_mean')
        import open3d as o3d
        from draw_pcloud import drawn_image
        if TRACK_COLOR:
            pcd_main=pointcloud_open3d(self.pcloud_main['xyz'],self.pcloud_main['rgb'])
        else:
            pcd_main=pointcloud_open3d(self.pcloud_main['xyz'], None)
        pdb.set_trace()
        dI=drawn_image(pcd_main)
        boxes = [ obj_.box for obj_ in positive_clusters ]
        dI.add_boxes_to_fg(boxes)
        fName=f"draw_clusters.{self.pcloud_main['xyz'].shape[0]}.png"
        if self.storage_dir is not None:
            fName=self.storage_dir+"/"+fName
        dI.save_fg(fName)
        resp=TriggerResponse()
        resp.success=True
        resp.message=fName
        return resp

    def cam_info_callback(self, cam_info):
        # print("Cam info received")
        self.params=camera_params(cam_info.height, cam_info.width, cam_info.K[0], cam_info.K[4], cam_info.K[2], cam_info.K[5], np.identity(4,dtype=float))
        self.pcloud_creator=pcloud_from_images(self.params)
        self.camera_params_sub.unregister()

    def pose_callback(self, odom_msg):
        # print("pose received")
        self.pose_lock.acquire()
        self.pose_queue.append(odom_msg)
        if len(self.pose_queue)>20:
            self.pose_queue.pop(0)
        self.pose_lock.release()

    def get_pose(self, tStamp):
        t=tStamp.to_nsec()
        self.pose_lock.acquire()
        top=None
        bottom=None
        for count, value in enumerate(self.pose_queue):
            if value.header.stamp.to_nsec()>t:
                top=value
                if count>0:
                    bottom=self.pose_queue[count-1]
                break
        self.pose_lock.release()
        if top is None or bottom is None:
            return None
        # Linear Interpolation between timestamps
        slopeT=(t-bottom.header.stamp.to_nsec())/(top.header.stamp.to_nsec()-bottom.header.stamp.to_nsec())
        topP=np.array([top.pose.pose.position.x,top.pose.pose.position.y,top.pose.pose.position.z])
        bottomP=np.array([bottom.pose.pose.position.x,bottom.pose.pose.position.y,bottom.pose.pose.position.z])
        pose = bottomP + slopeT*(topP-bottomP)

        # Also need to calculate orientation - interpolation between euler angles
        topQ=[top.pose.pose.orientation.x, top.pose.pose.orientation.y, top.pose.pose.orientation.z, top.pose.pose.orientation.w]
        bottomQ=[bottom.pose.pose.orientation.x, bottom.pose.pose.orientation.y, bottom.pose.pose.orientation.z, top.pose.pose.orientation.w]
        [a1,b1,topYaw]=tf.transformations.euler_from_quaternion(topQ)
        [a2,b2,bottomYaw]=tf.transformations.euler_from_quaternion(bottomQ)
        # We are going to assume that the shortest delta is the direction of rotation
        deltaY=normalizeAngle(topYaw-bottomYaw)
        if deltaY>np.pi:
            deltaY=2*np.pi-deltaY
        if deltaY<-np.pi:
            deltaY=2*np.pi + deltaY
        orientation=bottomYaw+deltaY*slopeT
        poseM=tf.transformations.rotation_matrix(orientation,(0,0,1))
        poseM[:3,3]=pose
        return poseM

    def is_new_image(self, new_poseM):
        if self.last_image is None:
            return True
        
        deltaP=self.last_image['poseM'][:,3]-new_poseM[:,3]
        dist=np.sqrt((deltaP**2).sum())
        vec1=np.matmul(self.last_image['poseM'][:3,:3],[1,0,0])
        vec2=np.matmul(new_poseM[:3,:3],[1,0,0])
        deltaAngle=np.arccos(np.dot(vec1,vec2))
        if dist>self.travel_params[0] or deltaAngle>self.travel_params[1]:
            print(f"Dist {dist}, angle {deltaAngle} - new")        
            return True
        return False

    def rgbd_callback(self, rgb_img, depth_img):
        print("RGB-D images received")
        if self.pcloud_creator is None:
            return
        if 0: # if the /map transform is correctly setup, then use tf all the way
            try:
                (trans,rot) = self.listener.lookupTransform('/map',depth_img.header.frame_id,depth_img.header.stamp)
                poseM=np.matmul(tf.transformations.translation_matrix(trans),tf.transformations.quaternion_matrix(rot))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("No Transform found")
                return
        else: # otherwise just get the transform to the base and then use the published odometry ... less accurate
            try:
                (trans,rot) = self.listener.lookupTransform('/base_link',depth_img.header.frame_id,depth_img.header.stamp)
                base_relativeM=np.matmul(tf.transformations.translation_matrix(trans),tf.transformations.quaternion_matrix(rot))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("No Transform found")
                return
            odom=self.get_pose(rgb_img.header.stamp)
            if odom is None:
                print("Missing odometry information - skipping")
                return
            
            # Convert the ROS Image message to a CV2 Image
            poseM=np.matmul(odom,base_relativeM)
        
        try:
            cv_image_rgb = self.bridge.imgmsg_to_cv2(rgb_img, "bgr8")
            cv_image_depth = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))    
            return

        if self.is_new_image(poseM):
            self.update_point_cloud(cv_image_rgb, cv_image_depth, poseM)
    
    def update_point_cloud(self, rgb, depth, poseM):
        self.last_image={'depth': rgb, 'rgb': depth, 'poseM': poseM}
        uid_key="%0.3f_%0.3f_%0.3f"%(poseM[0][3],poseM[1][3],poseM[2][3])
        self.pcloud_creator.load_image(rgb, depth, poseM, uid_key=uid_key)
        if self.storage_dir is not None:
            cv2.imwrite(self.storage_dir+"/rgb"+uid_key+".png",rgb)
            cv2.imwrite(self.storage_dir+"/depth"+uid_key+".png",rgb)
        results=self.pcloud_creator.multi_prompt_process([self.query_main, self.query_llm], self.detection_threshold, rotate90=True)
        if results[self.query_main]['xyz'].shape[0]>0:
            self.pcloud_main['xyz']=np.vstack((self.pcloud_main['xyz'],results[self.query_main]['xyz']))
            self.pcloud_main['probs']=np.hstack((self.pcloud_main['probs'],results[self.query_main]['probs']))
            if TRACK_COLOR:
                self.pcloud_main['rgb']=np.vstack((self.pcloud_main['rgb'],results[self.query_main]['rgb']))

        if results[self.query_llm]['xyz'].shape[0]>0:
            self.pcloud_llm['xyz']=np.vstack((self.pcloud_llm['xyz'],results[self.query_llm]['xyz']))
            self.pcloud_llm['probs']=np.hstack((self.pcloud_llm['probs'],results[self.query_llm]['probs']))
            if TRACK_COLOR:
                self.pcloud_llm['rgb']=np.vstack((self.pcloud_llm['rgb'],results[self.query_llm]['rgb']))
        print(f"Adding main:{results[self.query_main]['xyz'].shape[0]}, llm:{results[self.query_llm]['xyz'].shape[0]}.... Totals main:{self.pcloud_main['xyz'].shape[0]}, llm:{self.pcloud_llm['xyz'].shape[0]}")

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('main_target', type=str, help='main target search string')
    parser.add_argument('llm_target', type=str, help='llm target search string')
    parser.add_argument('--num_points',type=int,default=200, help='number of points per cluster')
    parser.add_argument('--detection_threshold',type=float,default=0.5, help='fixed detection threshold')
    parser.add_argument('--min_travel_dist',type=float,default=0.1,help='Minimum distance the robot must travel before adding a new image to the point cloud (default = 0.1m)')
    parser.add_argument('--min_travel_angle',type=float,default=0.1,help='Minimum angle the camera must have moved before adding a new image to the point cloud (default = 0.1 rad)')
    parser.add_argument('--storage_dir',type=str,default=None,help='A place to store intermediate files - but only if specified (default = None)')
    args = parser.parse_args()

    IT=two_query_localize(args.main_target,
                          args.llm_target,
                          args.num_points,
                          args.detection_threshold,
                          [args.min_travel_dist,args.min_travel_angle],
                          args.storage_dir)
    rospy.spin()