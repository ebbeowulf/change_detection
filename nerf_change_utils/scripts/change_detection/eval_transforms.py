import json
import pdb

fname="/data3/datasets/garden/fold3/3_4_5/transforms.json"

with open(fname,'r') as A:
    tf=json.load(A)

# sort
all_poses=dict()
for frame in tf['frames']:
    fpath=frame['file_path']
    try:
        frame_id=int(fpath.split('_')[-1].split('.')[0])
        all_poses[frame_id]=frame['transform_matrix']
    except Exception as e:
        pdb.set_trace()

pdb.set_trace()