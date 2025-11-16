#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from change_pcloud_utils.camera_params import camera_params
# from change_pcloud_utils.map_utils import pcloud_from_images, create_object_clusters, calculate_iou
from change_pcloud_utils.colmap_to_labeled_change_images import setup_change_experiment
from change_pcloud_utils.pcloud_creation_utils import build_pclouds
from stretch_srvs.srv import GetCluster

class change_server(Node):
    def __init__(self,
                 exp_params):
        super().__init__('change_server')
        
        self.exp_params=exp_params

        # To speed up processing, build the point clouds using
        #   colmap_to_labeled_change_images 
        self.pcloud_fNames = build_pclouds(exp_params['fList_new'],
                    exp_params['fList_renders'],
                    exp_params['prompts'],
                    exp_params['params'],
                    exp_params['detection_threshold'],
                    rebuild_pcloud=False)

        self.top1_cluster_srv = self.create_service(GetCluster, 'get_top1_cluster', self.top1_cluster_service)

    def top1_cluster_service(self, request, response):
        print("Received request for top1 cluster for prompt: " + request.main_query)
        if request.main_query not in self.exp_params['prompts']:
            print("Prompt not found in experiment prompts")
            response.success=False
            return response
        response.success=False
        response.message="method not implemented yet"
        return response

def main():
    exp_params=setup_change_experiment()

    rclpy.init() 

    change_srv=change_server(exp_params)
    rclpy.spin(change_srv) 

    change_srv.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': 
    main()
