from nerfstudio.process_data.colmap_utils import colmap_to_json, create_ply_from_colmap
from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('colmap_path',type=str,help="path to colmap binaries")
parser.add_argument('output_dir',type=str,help="location to place 2 files: transforms.json, transforms_export.json")
args = parser.parse_args()

recon=Path(args.colmap_path)
outDir=Path(args.output_dir)
res=colmap_to_json(recon,outDir)
if res<1:
    print("Failed to create transform - exiting")
    exit(-1)
fName=args.output_dir+"/transforms.json"

import numpy as np
import torch
applied_transform = np.eye(4)[:3, :]
applied_transform = applied_transform[np.array([0, 2, 1]), :]
applied_transform[2, :] *= -1

create_ply_from_colmap('sparse_pc.ply', recon, outDir, torch.from_numpy(applied_transform).float())
