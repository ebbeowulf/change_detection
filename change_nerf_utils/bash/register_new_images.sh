#!/bin/bash

# Before running this, need to run create_initial_dir-rgbd.sh to build the default colmap pipeline

#Check if CHANGE_HOME is set
set -euo pipefail
source is_home_set.sh
PYTHON_HOME=$CHANGE_HOME/change_nerf_utils/src/change_nerf_utils


INITIAL_DIR=$1 #this is the nerf data directory that contains colmap/
NEW_DIR_ROOT=$2 #this is the root directory that should contain color/ and depth/ subdirs

COLMAP_DIR=${NEW_DIR_ROOT}/colmap_combined
IMAGE_DIR=${NEW_DIR_ROOT}/images_combined
OLD_POSE_FILE=${INITIAL_DIR}/../camera_pose.txt

#This needs to be changed to point to your vocab tree file - which can be downloaded from
#      https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words1M.bin
VOCAB_TREE_VERSION="1M" #options are 32K, 256K, 1M
VOCAB_TREE="${CHANGE_HOME}/data/vocab_tree_flickr100K_words${VOCAB_TREE_VERSION}.bin"
if [[ ! -f $VOCAB_TREE ]]; then
    mkdir -p ${CHANGE_HOME}/data
    echo "VOCAB_TREE file not found at $VOCAB_TREE - downloading"
    LOCAL_VOCAB_FILE="vocab_tree_flickr100K_words${VOCAB_TREE_VERSION}.bin"
    cmd="wget https://github.com/colmap/colmap/releases/download/3.11.1/$LOCAL_VOCAB_FILE -O $VOCAB_TREE"
    echo $cmd
    eval $cmd
fi

# Convert the initial pose file into a format COLMAP can use
NEW_POSE_FILE=$NEW_DIR_ROOT/camera_pose.txt
if [[ ! -f $NEW_POSE_FILE ]]; then
    cmd="python ${PYTHON_HOME}/generate_initial_poses.py $NEW_DIR_ROOT/poses.csv $NEW_POSE_FILE"
    echo $cmd
    eval $cmd
fi

if [[ ! -f ${COLMAP_DIR}/database.db ]]; then
    echo "Step 1 - Preparing directory"
    echo "./prepare_dir.sh $INITIAL_DIR $NEW_DIR_ROOT"
    ./prepare_dir.sh $INITIAL_DIR $NEW_DIR_ROOT
fi

# Register the new images with the existing model
SPARSE_0=${COLMAP_DIR}/sparse_combined/0
if [[ ! -f $SPARSE_0/images.bin ]]; then
    echo "Step 2 - Extract features"
    NEW_IMAGES=$NEW_DIR_ROOT/new_images.txt
    cmd="colmap feature_extractor --database_path ${COLMAP_DIR}/database.db --image_path ${IMAGE_DIR} --image_list_path ${NEW_IMAGES} --ImageReader.single_camera 1 --ImageReader.existing_camera_id 1 --SiftExtraction.use_gpu 1"
    echo $cmd
    eval $cmd

    echo "Step 3 - Match features with existing images"
    cmd="colmap vocab_tree_matcher --database_path ${COLMAP_DIR}/database.db --VocabTreeMatching.vocab_tree_path $VOCAB_TREE --SiftMatching.use_gpu 1"
    # cmd="colmap exhaustive_matcher --database_path ${COLMAP_DIR}/database.db --SiftMatching.use_gpu 1"
    echo $cmd
    eval $cmd

    echo "Step 4 - Register images between old and new"
    mkdir -p ${SPARSE_0}
    cmd="colmap image_registrator --database_path ${COLMAP_DIR}/database.db --input_path ${INITIAL_DIR}/colmap/sparse/0/ --output_path ${SPARSE_0}"
    echo $cmd
    eval $cmd

    echo "Step 5 - Re-run bundle adjustment"
    cmd="colmap bundle_adjuster --input_path ${SPARSE_0} --output_path ${SPARSE_0} --BundleAdjustment.refine_principal_point 1"
    echo $cmd
    eval $cmd
fi

SPARSE_GEO_0=${COLMAP_DIR}/sparse_geo/0
if [[ ! -f ${SPARSE_GEO_0}/images.txt ]]; then
    mkdir -p ${SPARSE_GEO_0}
    echo "Step 6 - Align model to robot poses"
    cmd="colmap model_aligner --input_path ${SPARSE_0} --output_path ${SPARSE_GEO_0} --alignment_max_error 1 --ref_is_gps 0 --ref_images_path ${OLD_POSE_FILE}"
    echo $cmd
    eval $cmd

    echo "Step 7 - Convert to txt format"
    cmd="colmap model_converter --input_path ${SPARSE_GEO_0} --output_path ${SPARSE_GEO_0} --output_type TXT"
    echo $cmd
    eval $cmd

    echo "Step 8 - Convert to transforms.json"
    cmd="python ${PYTHON_HOME}/colmap_to_json.py ${SPARSE_GEO_0} ${NEW_DIR_ROOT}"
    echo $cmd
    eval $cmd
fi

echo ""
echo "To generate images, now run:"
echo ""
echo "./generate_nerfstudio_image.sh {splatfacto/depthacto output dir} ${NEW_DIR_ROOT} ${NEW_DIR_ROOT}/renders"
echo ""
