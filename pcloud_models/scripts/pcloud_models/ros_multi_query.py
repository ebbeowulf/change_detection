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
from msg_and_srv.srv import DynamicCluster, DynamicClusterRequest, DynamicClusterResponse, SetInt, SetIntRequest, SetIntResponse
from msg_and_srv.srv import SetString, SetStringResponse, SetStringRequest, SetFloat, SetFloatResponse, SetFloatRequest
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import torch
import transformers


TRACK_COLOR=True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalizeAngle(angle):
    while angle>=np.pi:
        angle-=2*np.pi
    while angle<-np.pi:
        angle+=2*np.pi
    return angle

class multi_query_localize:
    def __init__(self, query_list, classifier, cluster_min_points, detection_threshold, travel_params, storage_dir=None):
        # Initization of the node, name_sub
        rospy.init_node('multi_query_localize', anonymous=True)
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

        # Previously detected clusters
        self.known_objects=[]

        # Create Pcloud Data Structures
        self.pcloud_creator=None
        self.query_list=query_list
        self.classifier=classifier
        self.pcloud=dict()
        for query in self.query_list:
            # self.pcloud[query]={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
            self.pcloud[query]={'xyz': torch.zeros((0,3),dtype=float,device=DEVICE), 'probs': torch.zeros((0),dtype=float,device=DEVICE), 'rgb': torch.zeros((0,3),dtype=float,device=DEVICE)}
        self.cluster_min_points=cluster_min_points
        self.cluster_iou=0.1
        self.detection_threshold=detection_threshold
        self.cluster_metric='combo-mean'
        self.gridcell_size=0.01

        # Setup callback function
        self.camera_params_sub = rospy.Subscriber('/camera_throttled/depth/camera_info', CameraInfo, self.cam_info_callback)
        self.rgb_sub = message_filters.Subscriber('/camera_throttled/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera_throttled/depth/image_rect_raw', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.rgbd_callback)

        # Setup service calls
        self.setClusterMetric_srv = rospy.Service('set_cluster_metric', SetString, self.set_cluster_metric_service)
        self.setclustersize_srv = rospy.Service('set_cluster_size', SetInt, self.set_cluster_size_service)
        self.setiou_srv = rospy.Service('set_cluster_iou_pct', SetInt, self.set_cluster_iou_pct)
        self.setGridcell_srv = rospy.Service('set_gridcell_size', SetFloat, self.set_gridcell_size_service)
        self.clear_srv = rospy.Service('clear_clouds', Trigger, self.clear_clouds_service)
        self.top1_cluster_srv = rospy.Service('get_top1_cluster', DynamicCluster, self.top1_cluster_service)
        self.marker_pub=rospy.Publisher('cluster_markers',MarkerArray,queue_size=5)
        self.draw_clusters_srv = rospy.Service('draw_pclouds', Trigger, self.draw_pclouds)
    def draw_pclouds(self, msg):

        # positive_clusters, pos_likelihoods = self.match_clusters('combo-mean')
        import open3d as o3d
        from draw_pcloud import drawn_image

        for query in self.query_list:
            if self.pcloud[query]['xyz'].shape[0]>0:
                positive_clusters, pos_likelihoods=self.match_clusters(self.cluster_metric,query, query)
                if TRACK_COLOR:
                    pcd_main=pointcloud_open3d(self.pcloud[query]['xyz'].cpu().numpy(),self.pcloud[query]['rgb'].cpu().numpy())
                else:
                    pcd_main=pointcloud_open3d(self.pcloud[query]['xyz'].cpu().numpy(), None)
                
                dI=drawn_image(pcd_main)
                boxes = [ obj_.box for obj_ in positive_clusters ]
                dI.add_boxes_to_fg(boxes)
                fName=f"draw_clusters.{query.replace(' ','_')}.{self.pcloud[query]['xyz'].shape[0]}.png"
                pName=f"draw_clusters.{query.replace(' ','_')}.{self.pcloud[query]['xyz'].shape[0]}.ply"
                if self.storage_dir is not None:
                    fName=self.storage_dir+"/"+fName
                dI.save_fg(fName)
                o3d.io.write_point_cloud(pName, pcd_main)

        resp=TriggerResponse()
        resp.success=True
        resp.message=fName
        return resp

    def set_cluster_metric_service(self, req):
        if req.value=='main-mean':
            print("Setting new cluster metric as main-mean")
            self.cluster_metric='main-mean'
        elif req.value=='combo-mean': # combined mean
            print("Setting new cluster metric as combo-mean")
            self.cluster_metric='combo-mean'
        elif req.value=='main-room': # combined main + room
            print("Setting new cluster metric as main-room")
            self.cluster_metric='main-room'
        elif req.value=='combo-room': # combined main + room
            print("Setting new cluster metric as combo-room")
            self.cluster_metric='combo-room'
        elif req.value=='hybrid-boost': # special hybrid metric
            print("Setting new cluster metric as hybrid-boost")
            self.cluster_metric='hybrid-boost'
        elif req.value=='main-omdet': # OmDet only
            print("Setting new cluster metric as main-omdet")
            self.cluster_metric='main-omdet'
        elif req.value=='main-clipseg': # CLIPSeg only
            print("Setting new cluster metric as main-clipseg")
            self.cluster_metric='main-clipseg'
        else:
            print('Currently supported options include main-mean, combo-mean, main-room, combo-room, hybrid-boost, main-omdet, main-clipseg')
        return SetStringResponse()

    def set_cluster_size_service(self, req):
        self.cluster_min_points=req.value
        print(f"Changing minimum cluster size to {self.cluster_min_points} cm2")
        return SetIntResponse()

    def set_gridcell_size_service(self, req):
        self.gridcell_size=req.value
        print(f"Changing gridcell size to {self.gridcell_size} m")
        return SetFloatResponse()

    def set_cluster_iou_pct(self, req):
        self.cluster_iou=req.value/100.0
        print(f"Changing minimum cluster iou to {self.cluster_iou} pct")
        return SetIntResponse()

    def clear_clouds_service(self, msg):
        resp=TriggerResponse()
        resp.success=True
        for query in query_list:
            # self.pcloud[query]={'xyz': np.zeros((0,3),dtype=float), 'probs': np.zeros((0),dtype=float), 'rgb': np.zeros((0,3),dtype=float)}
            self.pcloud[query]={'xyz': torch.zeros((0,3),dtype=float,device=DEVICE), 'probs': torch.zeros((0),dtype=float,device=DEVICE), 'rgb': torch.zeros((0,3),dtype=float,device=DEVICE)}
        self.last_image=None
        resp.message="clouds cleared"
        return resp

    # def print_clouds_service(self, msg):
    #     resp=TriggerResponse()
    #     resp.success=False
    #     if self.pcloud_creator is None:
    #         resp.message="Pcloud creator not initialized"
    #         return resp
    #     import open3d as o3d
    #     if TRACK_COLOR:
    #         pcd_main=pointcloud_open3d(self.pcloud_main['xyz'],self.pcloud_main['rgb'])
    #         pcd_llm=pointcloud_open3d(self.pcloud_llm['xyz'],self.pcloud_llm['rgb'])
    #     else:
    #         resp.message="Color from probability ... not implemented yet"
    #         return resp
    #     fileName_main=f"{self.query_main}.{self.pcloud_main['xyz'].shape[0]}.ply"
    #     fileName_llm=f"{self.query_llm}.{self.pcloud_llm['xyz'].shape[0]}.ply"
    #     if self.storage_dir:
    #         fileName_llm=self.storage_dir+"/"+fileName_llm
    #         fileName_main=self.storage_dir+"/"+fileName_main
    #     o3d.io.write_point_cloud(fileName_main,pcd_main)
    #     o3d.io.write_point_cloud(fileName_llm,pcd_llm)
    #     resp.success=True
    #     resp.message=f"{fileName_main},{fileName_llm}"
    #     return resp

    # create the object clusters and filter by the number of points in each
    def get_clusters_ros(self, pcd):
        all_objects=create_object_clusters(pcd['xyz'].cpu().numpy(),
                                            pcd['probs'].cpu().numpy(), 
                                            -1.0, 
                                            self.detection_threshold, 
                                            compress_clusters=False,
                                            gridcell_size=self.gridcell_size)
        objects_out=[]
        for obj in all_objects:
            if obj.size()>self.cluster_min_points:
                objects_out.append(obj)
        return objects_out

    def publish_object_markers(self):
        msg=MarkerArray()
        for idx, obj_ in enumerate(self.known_objects):
            M=Marker()
            M.header.frame_id="map"
            M.header.stamp=self.last_image['time']
            M.type=Marker.SPHERE
            M.action=Marker.ADD
            mn_=obj_[1].mean(0)
            M.id=idx
            M.pose.position.x=mn_[0]
            M.pose.position.y=mn_[1]
            M.pose.position.z=mn_[2]
            M.pose.orientation.x=0.0
            M.pose.orientation.y=0.0
            M.pose.orientation.z=0.0
            M.pose.orientation.w=1.0
            M.scale.x=0.5
            M.scale.y=0.5
            M.scale.z=0.5
            M.color.a=1.0
            if obj_[0]=='main':
                M.color.r=1.0
                M.color.g=0.0
                M.color.b=0.0
            elif obj_[0]=='llm':
                M.color.r=0.0
                M.color.g=0.0
                M.color.b=1.0
            else:
                M.color.r=0.0
                M.color.g=1.0
                M.color.b=0.0
            msg.markers.append(M)
        self.marker_pub.publish(msg)

    # create the clusters and match them, returning any that exceed the detection threshold
    def match_clusters(self, method, main_query, llm_query):
        # Handle model-specific methods
        if '-omdet' in method or '-clipseg' in method:
            model = method.split('-')[-1]  # Extract model name (omdet or clipseg)
            base_method = method.replace(f"-{model}", "")  # Get base method name
            
            # Get the appropriate keys for model-specific point clouds
            main_key = f"{main_query}_{model}"
            llm_key = f"{llm_query}_{model}"
            
            # Check if keys exist in the point cloud dictionary
            if main_key in self.pcloud:
                objects_main = self.get_clusters_ros(self.pcloud[main_key])
            else:
                objects_main = []
                
            if llm_key in self.pcloud:
                objects_llm = self.get_clusters_ros(self.pcloud[llm_key])
            else:
                objects_llm = []
                
            # Save original method and temporarily set to the base method
            original_method = method
            method = base_method  # Use the base method (without the model suffix)
        else:
            # Original code for standard methods
            if main_query in self.query_list:
                objects_main = self.get_clusters_ros(self.pcloud[main_query])
            else:
                objects_main = []
                
            if llm_query in self.query_list:
                objects_llm = self.get_clusters_ros(self.pcloud[llm_query])
            else:
                objects_llm = []
                
            original_method = method  # No change for standard methods

        # Update current object list so that markers are published correctly
        self.known_objects=[]
        for obj_ in objects_main:
            self.known_objects.append(['main',obj_.box])
        for obj_ in objects_llm:
            self.known_objects.append(['llm',obj_.box])

        print(f"Clustering.... main {len(objects_main)}, llm {len(objects_llm)}")
        positive_clusters=[]
        positive_cluster_likelihood=[]
        # Match with other clusters
        if method=='main':
            positive_clusters=objects_main
            positive_cluster_likelihood=[ obj_.prob_stats['mean'] for obj_ in objects_main ]
        elif method=='llm':
            positive_clusters=objects_llm
            positive_cluster_likelihood=[ obj_.prob_stats['mean'] for obj_ in objects_llm ]
        else:
            for idx0 in range(len(objects_main)):                    
                cl_stats=[idx0, objects_main[idx0].prob_stats['max'], objects_main[idx0].prob_stats['mean'], -1, -1]
                for idx1 in range(len(objects_llm)):
                    IOU=calculate_iou(objects_main[idx0].box[0],objects_main[idx0].box[1],objects_llm[idx1].box[0],objects_llm[idx1].box[1])
                    if IOU>self.cluster_iou:
                        cl_stats[3]=max(cl_stats[3],objects_llm[idx1].prob_stats['max'])
                        cl_stats[4]=max(cl_stats[4],objects_llm[idx1].prob_stats['mean'])

                lk=estimate_likelihood(cl_stats, method)
                if lk>self.detection_threshold:
                    positive_clusters.append(objects_main[idx0])
                    positive_cluster_likelihood.append(lk)

            for obj_ in positive_clusters:
                self.known_objects.append(['combo',obj_.box])

        # publish the markers
        self.publish_object_markers()

        return positive_clusters, positive_cluster_likelihood

    def top1_cluster_service(self, request:DynamicClusterRequest):
        resp=DynamicClusterResponse()
        resp.success=False

        positive_clusters, pos_likelihoods=self.match_clusters(self.cluster_metric, self.query_list[0], self.query_list[1])
        if len(positive_clusters)==0:
            resp.message="No clusters found"
            return resp

        whichC=np.argmax(pos_likelihoods)
        if whichC>=0:
            positive_clusters[whichC].sample_pcloud(request.num_points)
            for idx in range(request.num_points):
                fPx=positive_clusters[whichC].farthestP[idx]
                pt=Point()
                pt.x=positive_clusters[whichC].pts[fPx][0]
                pt.y=positive_clusters[whichC].pts[fPx][1]
                pt.z=positive_clusters[whichC].pts[fPx][2]
                resp.pts.append(pt)
        resp.bbox3d=np.hstack((positive_clusters[whichC].box[0],positive_clusters[whichC].box[1])).tolist()
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
        if len(self.pose_queue)>200:
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

    def rgbd_callback(self, rgb_img:Image, depth_img:Image):
        # print("RGB-D images received")
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
            self.update_point_cloud(cv_image_rgb, cv_image_depth, poseM, rgb_img.header)
        # self.update_point_cloud(cv_image_rgb, cv_image_depth, poseM, rgb_img.header)
    
    def update_point_cloud(self, rgb, depth, poseM, header):
        self.last_image={'depth': rgb, 'rgb': depth, 'poseM': poseM, 'time': header.stamp}
        uid_key="%0.3f_%0.3f_%0.3f"%(poseM[0][3],poseM[1][3],poseM[2][3])
        self.pcloud_creator.load_image(rgb, depth, poseM, uid_key=uid_key)
        if self.storage_dir is not None:
            cv2.imwrite(self.storage_dir+"/rgb"+uid_key+".png",rgb)
            cv2.imwrite(self.storage_dir+"/depth"+uid_key+".png",rgb)
        results=self.pcloud_creator.multi_prompt_process(self.query_list, self.detection_threshold, rotate90=True, classifier_type=self.classifier)
        for query in self.query_list:
            if results is not None and results[query] is not None and results[query]['xyz'].shape[0]>0:
                # self.pcloud[query]['xyz']=np.vstack((self.pcloud[query]['xyz'],results[query]['xyz']))
                # self.pcloud[query]['probs']=np.hstack((self.pcloud[query]['probs'],results[query]['probs']))
                # if TRACK_COLOR:
                #     self.pcloud[query]['rgb']=np.vstack((self.pcloud[query]['rgb'],results[query]['rgb']))
                self.pcloud[query]['xyz']=torch.vstack((self.pcloud[query]['xyz'],results[query]['xyz']))
                self.pcloud[query]['probs']=torch.hstack((self.pcloud[query]['probs'],results[query]['probs']))
                if TRACK_COLOR:
                    self.pcloud[query]['rgb']=torch.vstack((self.pcloud[query]['rgb'],results[query]['rgb']))
                print(f"Adding {query}:{results[query]['xyz'].shape[0]}.... Total:{self.pcloud[query]['xyz'].shape[0]}")

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('queries', type=str, nargs='*', default=None,
                    help='Set of target queries to build point clouds for - must include at least one')
    parser.add_argument('--classifier',type=str,default='clipseg', help='select from available detection algorithms {clipseg, yolo, yolo_world}')
    parser.add_argument('--num_points',type=int,default=200, help='number of points per cluster')
    parser.add_argument('--detection_threshold',type=float,default=0.2, help='fixed detection threshold')
    parser.add_argument('--min_travel_dist',type=float,default=0.01,help='Minimum distance the robot must travel before adding a new image to the point cloud (default = 0.1m)')
    parser.add_argument('--min_travel_angle',type=float,default=0.05,help='Minimum angle the camera must have moved before adding a new image to the point cloud (default = 0.1 rad)')
    parser.add_argument('--storage_dir',type=str,default=None,help='A place to store intermediate files - but only if specified (default = None)')
    args = parser.parse_args()

    if args.queries is None:
        print("Must include at least one target query to execute")
        import sys
        sys.exit(-1)

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )
    # Get the target object from command-line arguments
    target = args.queries

    # Define the messages for the task
    messages = [
        {"role": "system", "content": "You are an expert at providing alternate queries to improve zero shot object detection models without using their names."},
        {"role": "user", "content": f"Describe how '{target[0]}' looks without using the word '{target[0]}' in 5 words or less."},
    ]


    # Generate the alternate query
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    
    target.append(outputs[0]["generated_text"][-1]["content"])
    print(f"Target object: {target}")

    IT=multi_query_localize(target,
                            args.classifier,
                          args.num_points,
                          args.detection_threshold,
                          [args.min_travel_dist,args.min_travel_angle],
                          args.storage_dir)
                          
    rospy.spin()
