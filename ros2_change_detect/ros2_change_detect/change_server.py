#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from change_pcloud_utils.camera_params import camera_params
# from change_pcloud_utils.map_utils import pcloud_from_images, create_object_clusters, calculate_iou
from change_pcloud_utils.colmap_to_labeled_change_images import setup_change_experiment
from change_pcloud_utils.pcloud_creation_utils import build_pclouds
from change_pcloud_utils.pcloud_cluster_utils import create_and_merge_clusters, merge_by_bounding_box
from stretch_srvs.srv import GetCluster, DrawCluster
from geometry_msgs.msg import Point
import numpy as np

ABSOLUTE_MIN_CLUSTER_SIZE=100
GRIDCELL_SIZE=0.01

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

        self.clear_loaded_clusters()

        self.top1_cluster_srv = self.create_service(GetCluster, 'get_top1_cluster', self.top1_cluster_service)
        self.top1_cluster_srv = self.create_service(DrawCluster, 'draw_clusters', self.draw_clusters_service)

    def clear_loaded_clusters(self):
        self.loaded_clusters={'prompt': None, 'clusters': None, 'pcloud': None }

    def top1_cluster_service(self, request, response):
        print("Received request for top1 cluster for prompt: " + request.main_query)
        if request.main_query not in self.exp_params['prompts']:
            print("Prompt not found in experiment prompts")
            response.success=False
            return response
        self.build_clusters(request.main_query)

        if self.loaded_clusters['clusters'] is None or len(self.loaded_clusters['clusters'])==0:
            print("No clusters found for prompt")
            response.success=False
            return response

        #criterion options are: 'max', 'mean', 'median', 'pcount'
        elif request.criterion in ['max','mean','median','pcount']:
            val_array=np.array([ cl_.prob_stats[request.criterion] for cl_ in self.loaded_clusters['clusters']])
        elif request.criterion in ['stdev', 'entropy']: # lower is better
            val_array=np.array([ -cl_.prob_stats[request.criterion] for cl_ in self.loaded_clusters['clusters']])
        else:
            print("Unknown criterion specified")
            response.success=False
            return response

        top1_idx=np.argmax(val_array)
        response.bbox3d=self.loaded_clusters['clusters'][top1_idx].box.reshape((6,)).tolist()

        # sample points randomly for now
        if request.num_points < self.loaded_clusters['clusters'][top1_idx].pts.shape[0]:
            pts=self.loaded_clusters['clusters'][top1_idx].pts
            sampled_pts = pts[np.random.choice(pts.shape[0], request.num_points, replace=False)]
            for p_idx in range(sampled_pts.shape[0]):
                P=Point()
                P.x=sampled_pts[p_idx,0]
                P.y=sampled_pts[p_idx,1]
                P.z=sampled_pts[p_idx,2]
                response.pts.append(P)
            response.success=True
            response.message="Top1 cluster returned successfully"
        else:
            response.success=False
            response.message="Not enough points in top1 cluster to satisfy request"
        return response

    def build_clusters(self, prompt):
        import pickle
        import os
        if self.loaded_clusters['prompt']==prompt:
            return True
        
        try:
            pcloud_fileName=self.pcloud_fNames[prompt]
            with open(pcloud_fileName, 'rb') as handle:
                self.loaded_clusters['pcloud']=pickle.load(handle)
        except Exception as e:
            print(f"pcloud file {pcloud_fileName} not found")
            self.clear_loaded_clusters()
            return False

        # Rescale everything ... 
        if self.loaded_clusters['pcloud']['xyz'].shape[0]>ABSOLUTE_MIN_CLUSTER_SIZE:
            self.loaded_clusters['clusters']=create_and_merge_clusters(self.loaded_clusters['pcloud']['xyz'].cpu().numpy(), GRIDCELL_SIZE)
            self.loaded_clusters['clusters']=merge_by_bounding_box(self.loaded_clusters['clusters'], self.loaded_clusters['pcloud'], self.exp_params['fList_new'], self.exp_params['fList_renders'], self.exp_params['params'])        

            for idx in range(len(self.loaded_clusters['clusters'])):
                self.loaded_clusters['clusters'][idx].estimate_probability(self.loaded_clusters['pcloud']['xyz'],self.loaded_clusters['pcloud']['probs'])
            return True
        else:
            print("Not enough points in point cloud to form clusters")
        return False
    
    def draw_clusters_service(self, request, resp):
        resp=DrawCluster.Response()
        resp.success=False

        # Build clusters if needed
        print(f"Received request to draw cluster {request.which_cluster} for prompt: " + request.main_query)
        self.build_clusters(request.main_query)

        if self.loaded_clusters['pcloud'] is None:
            print("Point cloud failed to load for prompt")
            resp.success=False
            return resp

        from change_pcloud_utils.draw_pcloud import drawn_image
        from change_pcloud_utils.map_utils import pointcloud_open3d

        if request.which_cluster>=0 and request.which_cluster<len(self.loaded_clusters['clusters']):            
            filt=self.loaded_clusters['clusters'][request.which_cluster].filter_points_in_box(self.loaded_clusters['pcloud']['xyz'])
            pts=self.loaded_clusters['pcloud']['xyz'][filt].to('cpu').numpy()
            clr=self.loaded_clusters['pcloud']['rgb'][filt].to('cpu').numpy()
            pcd_main=pointcloud_open3d(pts,clr)

            dI=drawn_image(pcd_main)
            fName=f"draw_clusters.{self.loaded_clusters['pcloud']['xyz'].shape[0]}_cl{request.which_cluster}.png"
            fName=self.exp_params['fList_new'].intermediate_save_dir+'/'+fName
            dI.save_fg(fName)

        elif request.which_cluster<0: # if which_cluster<0, draw all clusters and bounding boxes
            pcd_main=pointcloud_open3d(self.loaded_clusters['pcloud']['xyz'].to('cpu').numpy(),self.loaded_clusters['pcloud']['rgb'].to('cpu').numpy())

            dI=drawn_image(pcd_main)
            boxes = [ obj_.box for obj_ in self.loaded_clusters['clusters'] ]
            dI.add_boxes_to_fg(boxes)
            fName=f"draw_clusters.{self.loaded_clusters['pcloud']['xyz'].shape[0]}.png"
            fName=self.exp_params['fList_new'].intermediate_save_dir+'/'+fName
            dI.save_fg(fName)

        print(f"Saved drawn clusters to {fName}")
        resp.success=True    
        return resp
    

def main():
    exp_params=setup_change_experiment()

    rclpy.init() 

    change_srv=change_server(exp_params)
    rclpy.spin(change_srv) 

    change_srv.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__': 
    main()
