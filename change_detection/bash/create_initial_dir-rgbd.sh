#!/bin/bash

CHANGE_HOME=/home/emartinso/ros_ws/src/research/change_detection
BASH_HOME=$CHANGE_HOME/bash
PYTHON_HOME=$CHANGE_HOME/scripts/change_detection                                                                                                                                                         
BASE_DIR=$1
ROTATE_IMAGES=${2:-0}
COLMAP_NERF_DIR=$BASE_DIR/nerf_colmap

echo "ROTATE_IMAGES = $ROTATE_IMAGES"

if [[ $ROTATE_IMAGES -ne 0 ]]; then
    echo "Step 0 - Rotate images by 90 degrees"
    ROTATED_COLOR_DIR=${BASE_DIR}/color_rotated
    if [[ ! -d $ROTATED_COLOR_DIR ]]; then
        mkdir $ROTATED_COLOR_DIR
        bash $BASH_HOME/rotate_images.sh ${BASE_DIR}/color $ROTATED_COLOR_DIR
    fi
    COLOR_IMAGE_DIR=$ROTATED_COLOR_DIR

    ROTATED_DEPTH_DIR=${BASE_DIR}/depth_rotated
    if [[ ! -d $ROTATED_DEPTH_DIR ]]; then
        mkdir $ROTATED_DEPTH_DIR
        bash $BASH_HOME/rotate_images.sh ${BASE_DIR}/depth $ROTATED_DEPTH_DIR
    fi
    DEPTH_IMAGE_DIR=$ROTATED_DEPTH_DIR
else
    COLOR_IMAGE_DIR=${BASE_DIR}/color
    DEPTH_IMAGE_DIR=${BASE_DIR}/depth
fi

# Create symbolic links for images in the colmap directory
rm $COLMAP_NERF_DIR/images
rm $COLMAP_NERF_DIR/depth
ln -s $DEPTH_IMAGE_DIR $COLMAP_NERF_DIR/depth
ln -s $COLOR_IMAGE_DIR $COLMAP_NERF_DIR/images

# Step 1: Convert the initial pose file into a format COLMAP can use
NEW_POSE_FILE=$BASE_DIR/camera_pose.txt
cmd="python ${PYTHON_HOME}/generate_initial_poses.py $BASE_DIR/poses.csv $NEW_POSE_FILE"
echo $cmd
eval $cmd

# Step 2: Run the image registration step using the built-in nerfstudio tool
cd $BASE_DIR/
SPARSE=$COLMAP_NERF_DIR/colmap/sparse_orig
ln -s $SPARSE $COLMAP_NERF_DIR/colmap/sparse # temporary link for processing - will be removed later
if [[ ! -f $SPARSE/0/images.bin ]];then
    # cmd="ns-process-data images --data $COLOR_IMAGE_DIR --output-dir $COLMAP_NERF_DIR --skip-image-processing"
    # echo $cmd
    # eval $cmd

    # Alternative method that loads data via colmap without the ns-process-data step
    COLMAP_DB=$COLMAP_NERF_DIR/colmap/database.db
    VOCAB_TREE="/data2/datasets/office/vocab_tree_flickr100K_words1M.bin"   # no longer supported with colmab 3.12
    # VOCAB_TREE="/data2/datasets/office/vocab_tree_flickr100K_words256K.bin"

    mkdir -p $SPARSE/0    
    cmd="colmap feature_extractor --database_path $COLMAP_DB --image_path $COLOR_IMAGE_DIR --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1 --ImageReader.camera_model OPENCV"
    echo $cmd
    eval $cmd

    cmd="colmap vocab_tree_matcher --database_path $COLMAP_DB --VocabTreeMatching.vocab_tree_path $VOCAB_TREE --SiftMatching.use_gpu 1"
    echo $cmd
    eval $cmd

    cmd="colmap mapper --database_path $COLMAP_DB --image_path $COLOR_IMAGE_DIR --output_path $SPARSE --Mapper.ba_global_function_tolerance=1e-6"
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

    # Convert the aligned COLMAP model to transforms.json and sparse_pc.ply - copy these to the nerf_colmap directory if
    #   you want to use them during training or with depth images
    cmd="python $PYTHON_HOME/colmap_to_json.py $SPARSE_GEO/0 $SPARSE_GEO/0/"
    echo $cmd
    eval $cmd

    # Now, add depth image paths to the transforms.json file
    cmd="python $PYTHON_HOME/add_depth_to_transforms.py $SPARSE_GEO/0/transforms.json $SPARSE_GEO/0/transforms_with_depth.json"
    echo $cmd
    eval $cmd
fi

# Step 4: Need to prepare for using depth images - means copying the sparse_geo files to the home directory
#           and setting up symbolic links for the colmap/sparse directories
if [[ -f $SPARSE_GEO/0/transforms.json ]];then 
    rm -rf $COLMAP_NERF_DIR/colmap/sparse
    ln -s $SPARSE_GEO $COLMAP_NERF_DIR/colmap/sparse
    cp $SPARSE_GEO/0/transforms_with_depth.json $COLMAP_NERF_DIR/transforms.json
    cp $SPARSE_GEO/0/sparse_pc.ply $COLMAP_NERF_DIR/sparse_pc.ply
fi

# Step 5: Run nerfstudio
cd $BASE_DIR/
rm -rf $COLMAP_NERF_DIR/images
ln -s $COLOR_IMAGE_DIR $COLMAP_NERF_DIR/images
cmd="ns-train splatfacto --data nerf_colmap"
echo "Run one of the following commands from the $BASE_DIR to start training:"
echo "1) ns-train splatfacto --data nerf_colmap"
echo "2) ns-train depth-nerfacto --data nerf_colmap"
