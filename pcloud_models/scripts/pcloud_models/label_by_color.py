import argparse
import open3d as o3d
from rgbd_file_list import rgbd_file_list
import pdb
import numpy as np
from map_utils import pointcloud_open3d
import os
from map_utils import get_distinct_clusters
from draw_pcloud import drawn_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_ply',type=str,help='location of the labeled ply file to process')
    parser.add_argument('label', type=str, help='object type being labeled')
    parser.add_argument('color', type=str, help='color in string format \"red, blue, green\"')
    parser.add_argument('--annotation_file', type=str, default=None, help='where to store any resulting annotations')
    parser.add_argument('--min_cluster_points', type=int, default=200, help='minimum cluster size by point counts')
    args = parser.parse_args()

    pcd=o3d.io.read_point_cloud(args.labeled_ply)
    color=np.array([int(A) for A in args.color.split(',')])/255.0

    pt_xyz=np.array(pcd.points)
    pt_color=np.array(pcd.colors)
    matchP=np.where(np.sqrt(((pt_color-color)**2).sum(1)<0.05))
    pcd_small=pointcloud_open3d(pt_xyz[matchP],pt_color[matchP])
    # o3d.visualization.draw_geometries([pcd_small])
    object_clusters=get_distinct_clusters(pcd_small, cluster_min_count=args.min_cluster_points, gridcell_size=-1, eps=0.2)
    boxes = [ obj_.box for obj_ in object_clusters ]
    pdb.set_trace()

    if args.annotation_file is not None:
        # Then load and update the annotation file - replace existing entries
        annot=dict()
        if os.path.exists(args.annotation_file):
            import json
            with open(args.annotation_file,'r') as fin:
                annot=json.load(fin)
        pdb.set_trace()        

    dI=drawn_image(pcd)
    dI.add_boxes_to_fg(boxes)
    dI.draw_fg()
    pdb.set_trace()


