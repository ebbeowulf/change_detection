#!/bin/bash

# BASE_NERFSTUDIO_DIR="/home/emartinso/projects/nerfstudio/"
# CONFIG_DIR="outputs/nerf_no_person_initial/nerfacto/2024-01-04_035029/"
# SCRIPTS_DIR="$(pwd)/../scripts/"

CHANGE_HOME="$(pwd)/../"
SCRIPTS_DIR=${CHANGE_HOME}/scripts/change_detection

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
    cmd="python $SCRIPTS_DIR/colmap_to_json.py $TARGET_DIR/colmap_combined/sparse_geo/0 $TARGET_DIR"
    echo $cmd
    eval $cmd
fi

# Make the render save directory
if [[ ! -d $RENDER_SAVE_DIR ]];then
    mkdir -p $RENDER_SAVE_DIR
fi

# Render the image(s)
cmd="python $SCRIPTS_DIR/render_transform.py $CONFIG_DIR $transforms $RENDER_SAVE_DIR --image-type all --name-filter $IMAGE_NAME"
echo $cmd
eval $cmd

