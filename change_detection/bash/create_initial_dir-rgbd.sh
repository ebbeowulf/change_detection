#!/bin/bash

CHANGE_HOME=/home/emartinso/ros_ws/src/research/change_detection
BASH_HOME=$CHANGE_HOME/bash
PYTHON_HOME=$CHANGE_HOME/scripts/change_detection                                                                                                                                                         
BASE_DIR=$1
COLOR_IMAGE_DIR=$BASE_DIR/color
DEPTH_IMAGE_DIR=$BASE_DIR/depth

# Step 1: Convert the initial pose file into a format COLMAP can use
NEW_POSE_FILE=$BASE_DIR/camera_pose.txt
cmd="python ${PYTHON_HOME}/generate_initial_poses.py $BASE_DIR/poses.csv $NEW_POSE_FILE"
echo $cmd
eval $cmd

# Step 2: Run the image registration step using the built-in nerfstudio tool
cd $BASE_DIR/
COLMAP_NERF_DIR=$BASE_DIR/nerf_colmap
SPARSE=$COLMAP_NERF_DIR/colmap/sparse
if [[ ! -f $SPARSE/0/images.bin ]];then
    cmd="ns-process-data images --data $COLOR_IMAGE_DIR --output-dir $COLMAP_NERF_DIR --skip-image-processing"
    echo $cmd
    eval $cmd
fi

# Step 3: Align the COLMAP model to the initial poses
SPARSE_GEO=$COLMAP_NERF_DIR/colmap/sparse_geo
if [[ ! -f $SPARSE_GEO/0/images.bin ]];then
    mkdir -p $SPARSE_GEO/0
    cmd="colmap model_aligner --input_path $SPARSE/0 --output_path $SPARSE_GEO/0 --alignment_max_error 0.1 --ref_is_gps 0 --ref_images_path $NEW_POSE_FILE"
    echo $cmd
    eval $cmd

    cmd="colmap model_converter --input_path $SPARSE_GEO/0/ --output_path $SPARSE_GEO/0/ --output_type TXT"
    echo $cmd
    eval $cmd
fi

# Step 4: Run nerfstudio
cd $BASE_DIR/
rm -rf $COLMAP_NERF_DIR/images
ln -s $COLOR_IMAGE_DIR $COLMAP_NERF_DIR/images
cmd="ns-train splatfacto --data $COLMAP_NERF_DIR"
echo "Run the following command from $BASE_DIR to start training:"
echo $cmd
