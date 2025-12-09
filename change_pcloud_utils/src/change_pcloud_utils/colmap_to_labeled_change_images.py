# Builds the point clouds representing changes detected between two sets of images
#    and then clusters the changes and generates labeled images for each cluster
# Requires that the images have already been processed using the change_nerf_utils

import argparse
from change_pcloud_utils.pcloud_creation_utils import build_pclouds
from change_pcloud_utils.pcloud_cluster_utils import build_change_clusters, draw_boxes_around_cluster, write_change_cluster_images_to_disk
import numpy as np
from change_pcloud_utils.colmap_utils import get_camera_params, build_file_list, build_rendered_file_list
import subprocess
import pdb

def setup_change_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('root_dir',type=str,help='root project folder where the images and colmap info are stored')
    parser.add_argument('--color_dir',type=str,default='images_combined',help='where are the color images? (default=images_combined)')
    parser.add_argument('--renders_dir',type=str,default='renders',help='where are the rendered images? (default=renders)')
    parser.add_argument('--depth_dir',type=str,default='renders',help='where are the depth images? Use renders if nerfstudio generated. Or depth_rotated if from the robot (default=renders)')
    parser.add_argument('--colmap_dir',type=str,default='colmap_combined/sparse_combined/0',help='where are the images + cameras.txt files? (default=colmap_combined/sparse_combined/0)')
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
    parser.add_argument('--classifier',type=str,default='clipseg',help='Classifier type. Currently supports clipseg(default),grounded_dino,yolo_world,sam3')
    args = parser.parse_args()

    if args.use_change:
        save_dir=f"{args.root_dir}/{args.save_dir}/{args.classifier}_{args.threshold}_change"
    else:
        save_dir=f"{args.root_dir}/{args.save_dir}/{args.classifier}_{args.threshold}_openVocab"

    cmd = f'mkdir -p {save_dir}'
    subprocess.run(cmd, shell=True, check=True)
    
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
            'rebuild_pcloud': args.new_pcloud,
            'root_dir': args.root_dir,
            'classifier': args.classifier}

def clear_images(directory):
    # One-line bash command to delete common image types
    cmd = f'find "{directory}" -type f \\( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \\) -delete'    
    subprocess.run(cmd, shell=True, check=False)

def clear_clusters(directory):
    # One-line bash command to delete common image types
    cmd = f'find {directory}/*[0-9].pkl -delete'
    subprocess.run(cmd, shell=True, check=False)

def get_values_from_dict(d_in, tgt_key, default_val):
    p_array=[]
    for key in d_in:
        if tgt_key in d_in[key]:
            p_array.append(d_in[key][tgt_key])
        else:
            p_array.append(default_val)
    return p_array

if __name__ == '__main__':
    exp_params=setup_change_experiment()

    # We build the point clouds, but save to disk in order to 
    #   save memory for clustering
    pcloud_fNames = build_pclouds(exp_params['fList_new'],
                  exp_params['fList_renders'],
                  exp_params['prompts'],
                  exp_params['params'],
                  exp_params['detection_threshold'],
                  rebuild_pcloud=exp_params['rebuild_pcloud'],
                  classifier_type=exp_params['classifier'])

    clear_images(exp_params['fList_new'].intermediate_save_dir)
    clear_clusters(exp_params['fList_new'].intermediate_save_dir)
           
    for key in pcloud_fNames.keys():
        # Build the clusters
        pcloud, clusters = build_change_clusters(pcloud_fNames[key],
            exp_params['scale'])

        # Can apply additional merge opt at this point if so desired...
        # clusters=merge_by_bounding_box(clusters, pcloud, fList_new, fList_renders, params)
        from change_pcloud_utils.pcloud_cluster_utils import merge_by_bounding_box
        clusters=merge_by_bounding_box(clusters, pcloud, exp_params['fList_new'], exp_params['params'],-0.8)

        for cluster_idx, cluster in enumerate(clusters):
            all_images=draw_boxes_around_cluster(exp_params['fList_new'],
                                             exp_params['fList_renders'],
                                             exp_params['params'],
                                             pcloud,
                                             cluster)
            
            pn = np.array(get_values_from_dict(all_images, 'max_prob', exp_params['detection_threshold']))

            import pickle
            save_struct={'exp_params': exp_params,
                         'query': key,
                         'all_images': all_images,
                         'cluster': cluster}
            file_prefix=key.replace(' ','_')
            save_fName=exp_params['fList_new'].intermediate_save_dir+f"/{file_prefix}_{cluster_idx}.pkl"
            with open(save_fName,'wb') as handle:
                pickle.dump(save_struct, handle, protocol=pickle.HIGHEST_PROTOCOL)

            write_change_cluster_images_to_disk(all_images,exp_params['fList_new'],key,cluster_idx)

            # if pn.mean()>exp_params['detection_threshold']*1.2:
            #     write_change_cluster_images_to_disk(all_images,exp_params['fList_new'],key,cluster_idx)

            # at least 10% of the images looking at the target should be marked as
            #   containing change - this will impact objects that are only visible
            #   from one direction due to occlusions, but significantly cuts back
            #   on spurious false positives
            # valid_images=np.array([ 'new' in all_images[key] for key in all_images ])
            # if (valid_images.sum()/valid_images.shape[0])>0.2:
            #   write_change_cluster_images_to_disk(all_images,exp_params['fList_new'],key,cluster_idx)
            
            #   Filter clusters with LLM - comparing matching images
            # if cl_filter.evaluate_multiple_image_pairs(all_images):
            #     write_change_cluster_images_to_disk(all_images,exp_params['fList_new'],key,cluster_idx)

            #   Story to LLM - giving rules and trying to classify object type
            # if mv_filter.evaluate_cluster(all_images):
            #     write_change_cluster_images_to_disk(all_images,exp_params['fList_new'],key,cluster_idx)

            # Combine the before and after filter and a weak percentage filter
            # if ((valid_images.sum()/valid_images.shape[0])>0.1) and cl_filter.evaluate_multiple_image_pairs(all_images):
            #    write_change_cluster_images_to_disk(all_images,exp_params['fList_new'],key,cluster_idx)

            #   baseline - write everything
            # write_change_cluster_images_to_disk(all_images,exp_params['fList_new'],key,cluster_idx)

        # image_list=build_change_cluster_images(
        #     exp_params['fList_new'],
        #     exp_params['fList_renders'],
        #     exp_params['params'],
        #     pcloud_fNames[key],
        #     key)


