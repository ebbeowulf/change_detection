#!/bin/bash

BASE_NERFSTUDIO_DIR="/home/emartinso/projects/nerfstudio/"
CONFIG_DIR="outputs/nerf_no_person_initial/nerfacto/2024-01-04_035029/"
SCRIPTS_DIR=$(pwd)+"../scripts/"

TARGET_DIR=$1
RENDER_SAVE_DIR=$2
IMAGE_NUM=$3

#Initialize the conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash
conda activate nerfstudio3

cd $BASE_NERFSTUDIO_DIR

# Do we have the transforms.json file already?
transforms=$TARGET_DIR/transforms.json
if [[ ! -f $transforms ]];then
    python $SCRIPTS_DIR/colmap_to_json.py $TARGET_DIR/colmap_combined/sparse_geo $TARGET_DIR

# Have we created the nerf images?
rgb_fName="rgb_$IMAGE_NUM.png"
if [[ ! -f $RENDER_SAVE_DIR/$rgb_fName ]];then
    python $SCRIPTS_DIR/render_transform.py $CONFIG_DIR $transforms $RENDER_SAVE_DIR --name-filter $IMAGE_NUM --image-type all
fi

