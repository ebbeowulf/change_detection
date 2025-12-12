#!/usr/bin/env python3
import math
import time
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
import pdb

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Bool, Float64
from segmentation.dino_segmentation import dino_segmentation

import tf2_ros
from builtin_interfaces.msg import Time
from tf2_geometry_msgs import do_transform_point

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM
from std_srvs.srv import Trigger
from stretch_srvs.srv import SetDetectionTarget
from stretch_srvs.msg import ClusterArray, Cluster
from geometry_msgs.msg import Point

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISPLAY_RESULTS=True

# MULTI-TARGET LIST
TARGET_LABELS = [
    "trash",
    "electronics",
    "small items",
    "general clutter"
]

class CloseRangeDetector(Node):
    def __init__(self):
        super().__init__("close_range_detector")

        self.get_logger().info(f"Loaded multi-target list: {TARGET_LABELS}")

        self.dino=dino_segmentation(TARGET_LABELS)

        # ----------- CAMERA STATE ----------
        self.bridge = CvBridge()
        self.latest_depth = None
        self.has_depth = False
        self.fx = self.fy = self.cx = self.cy = None
        self.has_intrinsics = False
        self.camera_frame = "camera_color_optical_frame"

        # ----------- ROS SUB/PUB ----------
        self.create_subscription(Image, "/camera/color/image_raw", self.rgb_callback, 10)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.depth_callback, 10)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.caminfo_callback, 10)

        self.disable_detection_srv = self.create_service(Trigger, "/disable_detection",self.disable_detection_target_service)
        self.activate_detection_srv = self.create_service(SetDetectionTarget, "/set_detection_target",self.set_detection_target_service)
        self.cluster_pub = self.create_publisher(ClusterArray, "/DetectedObjects", 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.detection_target = None
        self.last_run_time = 0.0

        self.get_logger().info("Node 2 ready")
    
    # -------------------------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------------------------
    def set_detection_target_service(self, request, response):
        if self.dino.get_id(request.main_query) is None:
            # Need to change the prompt list
            self.dino.change_prompts([request.main_query]) 
        self.detection_target={'label': request.main_query, 'bbox3d': request.bbox3d, 'is_highest_confidence': request.report_highest_confidence}
        response.success=True
        return response

    def disable_detection_target_service(self, request, response):
        self.detection_target=None
        return response
        
    def caminfo_callback(self, msg):
        if not self.has_intrinsics:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.has_intrinsics = True
            self.get_logger().info("Camera intrinsics loaded.")

    def depth_callback(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.latest_depth = d.astype(np.float32) / 1000.0
        self.has_depth = True

    def display_results(self, label, pil_image, mask=None, bbox=None):
        if DISPLAY_RESULTS:
            overlay=np.array(pil_image)
            if mask is not None and bbox is not None:
                # ------ VISUALIZATION ------
                mask_vis = (mask.astype(np.uint8) * 255)
                mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(overlay, 0.6, mask_color, 0.4, 0)

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(overlay, f"{label}", (x1, y1-5),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(overlay, "NO OBJECT DETECTED", (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

            cv2.imshow("DINO-SAM Mask", overlay)
            cv2.waitKey(1)

    def estimate_best_match(self, pil_image):
        if self.detection_target is None:
            return None

        self.dino.process_image(pil_image,threshold=0.3)
        cls=self.detection_target['label']
        boxes=self.dino.get_boxes(cls)
        
        if boxes is None or len(boxes)==0:
            self.display_results(cls, pil_image)
            return None

        if self.detection_target['is_highest_confidence']:
            pdb.set_trace()
            conf_array=[ box[0] for box in boxes ]
            which_box = np.argmax(conf_array)
            sam_mask=self.dino.per_object_mask[self.dino.get_id(cls)][which_box].cpu().numpy()
            x,y,z=self.extract_touch_points(sam_mask, self.latest_depth,boxes[which_box][1])
            self.display_results(cls,pil_image,sam_mask,boxes[which_box][1])
            return [x,y,z]
        else:
            print("functionality not currently supported")
            return None
            # touchP=np.zeros((len(boxes),3),dtype=float)        
            # for idx, box in enumerate(boxes):
            #     sam_mask=self.dino.per_object_mask[self.dino.get_id(cls)][idx].cpu().numpy()
            #     Pt=self.extract_touch_points(sam_mask, self.latest_depth,box[1])
            #     if
            #     touchP[idx][0],touchP[idx][1],touchP[idx][2]=        

    def extract_touch_points(self, sam_mask, depth_img, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        mask_idx = np.where(sam_mask)

        depth_vals = depth_img[mask_idx]
        valid = (depth_vals > 0.2) & (depth_vals < 3.0)
        depth_vals = depth_vals[valid]

        if len(depth_vals) == 0:
            self.get_logger().warn("No valid depth inside mask, cannot estimate height.")
            return None

        # sample up to 500
        num = min(500, len(depth_vals))
        idx = np.random.choice(len(depth_vals), num, replace=False)
        sample = depth_vals[idx]

        Z = float(np.median(sample))
        self.get_logger().info(f"Median depth = {Z:.3f} m")

        # ------ PIXEL COORDINATES FOR PROJECTION ------
        u = int((x1 + x2)/2)
        v = int((y1 + y2)/2)

        Xc = (u - self.cx) * Z / self.fx
        Yc = (v - self.cy) * Z / self.fy
        return Xc, Yc, Z

    def publish_touch_point(self, header, X, Y, Z):
        p_cam = PointStamped()
        p_cam.header.frame_id = self.camera_frame
        p_cam.header.stamp = header.stamp
        p_cam.point.x = float(X)
        p_cam.point.y = float(Y)
        p_cam.point.z = float(Z)

        # ------ TRANSFORM TO BASE_LINK ------
        tf_cam_base = self.tf_buffer.lookup_transform(
            "base_link", self.camera_frame, Time()
        )
        p_base = do_transform_point(p_cam, tf_cam_base)
        self.get_logger().info(f"Object (base_link): x={p_base.point.x:.2f}, y={p_base.point.y:.2f}, z={p_base.point.z:.2f}")

        cl_=Cluster()
        cl_.obj_type=self.detection_target['label']
        cl_.pts.append(p_base.point)
        clA=ClusterArray()
        clA.header=header
        clA.all_clusters.append(cl_)
        self.cluster_pub.publish(clA)        

    # -------------------------------------------------------------------------
    # MAIN RGB CALLBACK
    # -------------------------------------------------------------------------
    def rgb_callback(self, msg):
        if not self.detection_target:
            return

        if not (self.has_intrinsics and self.has_depth):
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        H, W = frame.shape[:2]

        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        print("process_image")
        pt=self.estimate_best_match(pil_image)

        if pt is not None:
            self.publish_touch_point(msg.header,pt[0],pt[1],pt[2])
        else:
            print("No point found")
        self.detection_target=None

def main():    
    rclpy.init()
    node = CloseRangeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()