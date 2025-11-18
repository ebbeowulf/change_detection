#!/bin/bash

# BASE_NERFSTUDIO_DIR="/home/emartinso/projects/nerfstudio/"
# CONFIG_DIR="outputs/nerf_no_person_initial/nerfacto/2024-01-04_035029/"
# SCRIPTS_DIR="$(pwd)/../scripts/"

#Check if CHANGE_HOME is set
if ! source is_home_set.sh; then
    echo "Failed to source is_home_set.sh" >&2
    exit 1
fi

PYTHON_HOME=$CHANGE_HOME/change_nerf_utils/src/change_nerf_utils

BASE_CONFIG_DIR=$1
TARGET_DIR=$2
RENDER_SAVE_DIR=$3
IMAGE_NAME=${4:-new} #by default render all images with 'new' in the name

BASE_CONFIG_DIR=$1
delimiter="outputs"
BASE_NERFSTUDIO_DIR="${BASE_CONFIG_DIR%$delimiter*}"
CONFIG_DIR="${BASE_CONFIG_DIR#*$BASE_NERFSTUDIO_DIR}"

cd $BASE_NERFSTUDIO_DIR

# Do we have the transforms.json file already?
transforms=$TARGET_DIR/transforms.json
if [[ ! -f $transforms ]];then
    cmd="python $PYTHON_HOME/colmap_to_json.py $TARGET_DIR/colmap_combined/sparse_geo/0 $TARGET_DIR"
    echo $cmd
    eval $cmd
fi

# Make the render save directory
if [[ ! -d $RENDER_SAVE_DIR ]];then
    mkdir -p $RENDER_SAVE_DIR
fi

# Render the image(s)
cmd="python $PYTHON_HOME/render_transform.py $CONFIG_DIR $transforms $RENDER_SAVE_DIR --image-type all --name-filter $IMAGE_NAME"
echo $cmd
eval $cmd

