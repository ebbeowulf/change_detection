import os
import sys
import glob
from pcloud_models.object_pcloud_from_scannet import read_label_csv, string_from_id, get_scene_type
import pdb
import json

# SCANNET_DIR="/home/emartinso/data/scannet/scans/"
SCANNET_DIR="/data3/datasets/scannet/scans/"
SAVEFILE=SCANNET_DIR+"/object_in_room_stats.json"

# all_object_files = glob.glob(SCANNET_DIR+'/*/raw_output/save_results/all_labels.json')
all_scenes=glob.glob(SCANNET_DIR+"/scene*")

read_label_csv()

objects_per_scene={}
scene_per_object={}
room_type_per_scene={}

for root_dir in all_scenes:
    try:
        all_object_file=root_dir+"/raw_output/save_results/all_labels.json"
        all_objects={}
        if os.path.exists(all_object_file):
            with open(all_object_file,"r") as fin:
                all_objects=json.load(fin)
        else:
            continue

        #param file:
        s_root=root_dir.split('/')
        if s_root[-1]=='':
            par_file=root_dir+"%s.txt"%(s_root[-2])
        else:
            par_file=root_dir+"/%s.txt"%(s_root[-1])
        room_type=get_scene_type(par_file)
        # params=load_camera_info(par_file)

    except Exception as e:
        continue
    room_type_per_scene[root_dir]=room_type
    objects_per_scene[root_dir]=[]
    for oid in all_objects:
        object_type=string_from_id(int(oid))
        if object_type is None:
            continue
        object_type=object_type[0]
        objects_per_scene[root_dir].append(object_type)
        if object_type in scene_per_object:
            scene_per_object[object_type].append(root_dir)
        else:
            scene_per_object[object_type]=[root_dir]

all_objects= [ key for key in scene_per_object ]
print(all_objects)

combined_export={'objects_per_scene': objects_per_scene,
                 'scene_per_object': scene_per_object,
                 'room_type_per_scene': room_type_per_scene}

with open(SAVEFILE,"w") as fout:
    json.dump(combined_export, fout)

pdb.set_trace()