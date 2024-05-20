from nerfstudio.process_data.colmap_utils import colmap_to_json
from pathlib import Path
import json
import argparse
#recon=Path("/data2/datasets/home/dining_living_scan/colmap/sparse/1")
#outDir=Path("/data2/datasets/home/dining_living_scan/")
#recon=Path("/data2/datasets/home/register_messy_with_normal/colmap/sparse_combined/")
#outDir=Path("/data2/datasets/home/register_messy_with_normal/")
#recon=Path("/data2/datasets/office/registered_combined/colmap/sparse_combined/")
#outDir=Path("/data2/datasets/office/registered_combined/")

parser = argparse.ArgumentParser()
parser.add_argument('colmap_path',type=str,help="path to colmap binaries")
#parser.add_argument('export_file',type=str,help="result of ns-export - needed to determine new coordinate frame") 
parser.add_argument('output_dir',type=str,help="location to place 2 files: transforms.json, transforms_export.json")
args = parser.parse_args()

#recon=Path("/data2/datasets/office/1/colmap/sparse/0/")
#outDir=Path("/data2/datasets/office/1/")
recon=Path(args.colmap_path)
outDir=Path(args.output_dir)
res=colmap_to_json(recon,outDir)
if res<1:
    print("Failed to create transform - exiting")
    exit(-1)
fName=args.output_dir+"/transforms.json"
#with open(fName,'r') as fin:
#    TF_start=json.load(fin)
#with open(args.export_file,'r') as fin:
#    TF_export=json.load(fin)

