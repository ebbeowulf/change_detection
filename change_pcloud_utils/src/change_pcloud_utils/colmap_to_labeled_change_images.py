import argparse
from pcloud_creation_utils import build_pclouds
from pcloud_cluster_utils import build_change_cluster_images
import numpy as np
import os
import sys

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'change_detection', 'scripts', 'change_detection'))
sys.path.append(scripts_path)

from colmap_utils import get_camera_params, build_file_list, build_rendered_file_list


def setup_change_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('root_dir',type=str,help='root project folder where the images and colmap info are stored')
    parser.add_argument('--color_dir',type=str,default='images_combined',help='where are the color images? (default=images_combined)')
    parser.add_argument('--renders_dir',type=str,default='renders',help='where are the rendered images? (default=renders)')
    parser.add_argument('--depth_dir',type=str,default='renders',help='where are the depth images? Use renders if nerfstudio generated. Or depth_rotated if from the robot (default=renders)')
    parser.add_argument('--colmap_dir',type=str,default='colmap_combined/sparse_geo',help='where are the images + cameras.txt files? (default=colmap_combined/sparse)')
    parser.add_argument('--frame_keyword',type=str,default="new",help='a keyword to use when parsing the transforms file (default=new)')
    parser.add_argument('--save_dir',type=str,default='save_results', help='subdirectory of root_dir in which to store the intermediate files (default=save_results)')
    parser.add_argument('--queries', type=str, nargs='*', default=["General clutter", "Small items on surfaces", "Floor-level objects", "Decorative and functional items", "Trash items"],
                help='Set of target queries to build point clouds for - default is [General clutter, Small items on surfaces, Floor-level objects, Decorative and functional items, Trash items]')
    parser.add_argument('--threshold',type=float, default=0.3, help="fixed threshold to apply for change detection (default=0.3)")
    parser.add_argument('--max_route_dist',type=float,default=None, help='if using a phone, need to specify a known distance between end points in order to estimate scale (default = None, or use 5.2 for a typical room-scale scan)')
    parser.add_argument('--no-change', dest='use_change', action='store_false')
    parser.set_defaults(use_change=True)
    parser.add_argument('--rebuild-pcloud', dest='new_pcloud', action='store_true')
    parser.set_defaults(new_pcloud=False)
    args = parser.parse_args()

    save_dir=f"{args.root_dir}/{args.save_dir}/"
    color_image_dir=f"{args.root_dir}/{args.color_dir}/"
    depth_image_dir=f"{args.root_dir}/{args.depth_dir}/"
    rendered_image_dir=f"{args.root_dir}/{args.renders_dir}/"
    colmap_dir=f"{args.root_dir}/{args.colmap_dir}/"
    params=get_camera_params(colmap_dir,args.nerfacto_dir)
    fList_new=build_file_list(color_image_dir,depth_image_dir,save_dir,colmap_dir,args.frame_keyword)
    if args.use_change:
        fList_renders=build_rendered_file_list(fList_new, rendered_image_dir,save_dir)
    else:
        fList_renders=None
    if args.max_route_dist is None:
        scale=1.0
    else:
        # Need to estimate the scale factor from the known route distance
        allP=np.array([ fList_new.get_pose(key)[:3,3] for key in fList_new.keys()])
        first_pose=np.median(allP[:5,:],0)
        last_pose=np.median(allP[-5:,:],0)
        reconstructed_dist=np.sqrt(np.sum((last_pose-first_pose)**2))        
        scale=args.max_route_dist/reconstructed_dist

    prompts = [ s.lower() for s in args.queries ]
    return {'params': params,
            'fList_new': fList_new,
            'fList_renders': fList_renders,
            'detection_threshold': args.threshold,
            'prompts': prompts,
            'scale': scale,
            'rebuild_pcloud': args.new_pcloud}

if __name__ == '__main__':
    exp_params=setup_change_experiment()

    # We build the point clouds, but save to disk in order to 
    #   save memory for clustering
    pcloud_fNames = build_pclouds(exp_params['fList_new'],
                  exp_params['fList_renders'],
                  exp_params['prompts'],
                  exp_params['params'],
                  exp_params['detection_threshold'],
                  rebuild_pcloud=False)
        
    for key in pcloud_fNames.keys():
        image_list=build_change_cluster_images(
            exp_params['fList_new'],
            exp_params['fList_renders'],
            exp_params['params'],
            pcloud_fNames[key],
            key)


