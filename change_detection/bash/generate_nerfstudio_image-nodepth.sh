#!/bin/bash

# BASE_NERFSTUDIO_DIR="/home/emartinso/projects/nerfstudio/"
#CONFIG_DIR="outputs/nerf_plant/nerfacto/2024-06-27_025912/"
BASE_NERFSTUDIO_DIR="/data3/datasets/garden/"
CONFIG_DIR="outputs/garden_07_17_v1/nerfacto/2025-02-05_193728/"
SCRIPTS_DIR="$(pwd)/../scripts/change_detection/"

TARGET_DIR=$1
RENDER_SAVE_DIR=$2
RGB_FNAME=$3

#Initialize the conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda init bash
conda activate nerfstudio3

cd $BASE_NERFSTUDIO_DIR

# Do we have the transforms.json file already?
transforms=$TARGET_DIR/transforms.json
if [[ ! -f $transforms ]];then
    python $SCRIPTS_DIR/colmap_to_json.py $TARGET_DIR/colmap_combined/sparse_combined $TARGET_DIR
fi

# Have we created the nerf images?
#rgb_fName="rgb_$IMAGE_NUM.png"
if [[ ! -f $RENDER_SAVE_DIR/$RGB_FNAME ]];then
    if [[ ! -d $RENDER_SAVE_DIR ]];then
        mkdir -p $RENDER_SAVE_DIR
    fi
    echo "python $SCRIPTS_DIR/render_transform.py $CONFIG_DIR $transforms $RENDER_SAVE_DIR --name-filter $RGB_FNAME --image-type all"
    python $SCRIPTS_DIR/render_transform.py $CONFIG_DIR $transforms $RENDER_SAVE_DIR --name-filter $RGB_FNAME --image-type all
fi

