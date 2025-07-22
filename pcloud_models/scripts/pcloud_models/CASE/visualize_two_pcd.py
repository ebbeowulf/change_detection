import argparse
import open3d as o3d
import pickle
from map_utils import pointcloud_open3d
import os
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bg_pcloud', type=str, help='background pcloud - visualize with color + transparency')
    parser.add_argument('fg_pcloud', type=str, help='foreground pcloud - visualize with solid color')
    args = parser.parse_args()

    with open(args.bg_pcloud, 'rb') as handle:
        bg_pcloud=pickle.load(handle)
    
    bg_pcd_orig=pointcloud_open3d(bg_pcloud['xyz'],255*bg_pcloud['rgb'][:,[2,1,0]])
    bg_pcd, ind = bg_pcd_orig.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
    mat1 = o3d.visualization.rendering.MaterialRecord()
    mat1.shader = 'defaultLitTransparency'
    mat1.base_color = [1.0, 0.0, 0.0, 0.5]  # RGBA, adjust A for transparency
    mat1.point_size = 1

    with open(args.fg_pcloud, 'rb') as handle:
        fg_pcloud=pickle.load(handle)
    
    fg_pcd=pointcloud_open3d(fg_pcloud['xyz'].cpu().numpy())

    clouds = [{'name': 'background', 'geometry': bg_pcd, 'material': mat1},
          {'name': 'foreground', 'geometry': fg_pcd}]

    o3d.visualization.draw(clouds, bg_color=(0.1, 0.1, 0.1, 1.0), show_skybox=False)
    # o3d.visualization.draw([bg_pcd, fg_pcd])