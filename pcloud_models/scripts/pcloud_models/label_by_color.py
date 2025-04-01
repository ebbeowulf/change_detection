import argparse
import open3d as o3d
from rgbd_file_list import rgbd_file_list
import pdb
import numpy as np
from map_utils import pointcloud_open3d
import os
from map_utils import get_distinct_clusters
from draw_pcloud import drawn_image
from scannet_processing import load_camera_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of the root_directory')
    parser.add_argument('label', type=str, help='object type being labeled')
    parser.add_argument('color', type=str, help='color in string format \"red, blue, green\"')
    parser.add_argument('--min_cluster_points', type=int, default=200, help='minimum cluster size by point counts')
    parser.add_argument('--update_annotation', dest='annotate', action='store_true')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--color_threshold', type=float, default=0.002, help='distance in color space to accept pixels as belonging to the object')
    parser.set_defaults(annotate=False)

    args = parser.parse_args()
    image_dir=args.root_dir+"/"+args.raw_dir
    save_dir=args.root_dir+"/"+args.save_dir
    fList = rgbd_file_list(image_dir, image_dir, save_dir)
    s_root=args.root_dir.split('/')
    if s_root[-1]=='':
        par_file=args.root_dir+"%s.txt"%(s_root[-2])
    else:
        par_file=args.root_dir+"/%s.txt"%(s_root[-1])
    params=load_camera_info(par_file)

    if (params.rot_matrix== np.eye(4)).all():
        print(" ********************* ERROR ******************")
        print(" ******      Missing rotation matrix    *******")
        print(" ****** Cannot label the following file *******")
        print(f" ****** {args.root_dir} *******")
        print(" *********************************************")

    # Need to extract the labeled ply
    f_split=args.root_dir.split('/')
    labeled_ply=args.root_dir+'/'
    for idx in range(len(f_split)):
        if f_split[-(idx+1)]=='':
            continue
        labeled_ply+=f_split[-(idx+1)]+"_vh_clean_2.labels.ply"
        break

    pcd=o3d.io.read_point_cloud(labeled_ply)
    color=np.array([int(A) for A in args.color.split(',')])/255.0

    pts=np.array(pcd.points)
    pts4=np.ones((4,pts.shape[0]))
    pts4[:3,:]=pts.transpose()
    pts_rotated=np.matmul(params.rot_matrix,pts4).transpose()
    pcd.points=o3d.utility.Vector3dVector(pts_rotated[:,:3])
    pt_xyz=pts_rotated[:,:3]
    pt_color=np.array(pcd.colors)
    matchP=np.where(np.sqrt(((pt_color-color)**2).sum(1)<args.color_threshold))
    pcd_small=pointcloud_open3d(pt_xyz[matchP],pt_color[matchP])
    # o3d.visualization.draw_geometries([pcd_small])
    object_clusters=get_distinct_clusters(pcd_small, cluster_min_count=args.min_cluster_points, gridcell_size=-1, eps=0.2)
    boxes = [ obj_.box for obj_ in object_clusters ]
    print(f"Num objects found ... {len(boxes)}")
    if args.annotate:
        linear_boxes=[ np.hstack((box[0],box[1])).tolist() for box in boxes ]
        # for box in boxes:
        #     bb=np.ones((2,4),dtype=float)
        #     bb[:,:3]=box
        #     corners=np.matmul(params.rot_matrix, bb.transpose())
        #     rotated_boxes.append(np.hstack((corners[:3,0],corners[:3,1])).tolist())
        #     # rotated_boxes.append(

        # Then load and update the annotation file - replace existing entries
        annot=dict()
        annotation_file=fList.get_annotation_file()
        if os.path.exists(annotation_file):
            import json
            with open(annotation_file,'r') as fin:
                annot=json.load(fin)
        annot[args.label]=linear_boxes
        with open(annotation_file,'w') as fout:
            annot=json.dump(annot,fout)
    else:
        dI=drawn_image(pcd)
        dI.add_boxes_to_fg(boxes)
        dI.draw_fg()


