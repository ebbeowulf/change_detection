#!/bin/bash

# Before running this, need to run ns-process-data to build the default colmap pipeline

INITIAL_DIR=$1 #this is the nerf data directory that contains colmap/
NEW_DIR_ROOT=$2 #this is the root directory that should contain color/ and depth/ subdirs

COLMAP_DIR=${NEW_DIR_ROOT}/colmap_combined
IMAGE_DIR=${NEW_DIR_ROOT}/images_combined
NEW_POSE_FILE=$INITIAL_DIR/camera_pose.txt

#This needs to be changed to point to your vocab tree file - which can be downloaded from
#      https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words1M.bin
#VOCAB_TREE="/data2/datasets/office/vocab_tree_flickr100K_words256K.bin"
# VOCAB_TREE="/data2/datasets/office/vocab_tree_flickr100K_words32K.bin"
VOCAB_TREE="/data2/datasets/office/vocab_tree_flickr100K_words1M.bin"

echo "Step 1 - Preparing directory"
echo "./prepare_dir.sh $INITIAL_DIR $NEW_DIR_ROOT"
./prepare_dir.sh $INITIAL_DIR $NEW_DIR_ROOT

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
COLMAP_MODEL=${COLMAP_DIR}/sparse_combined
mkdir ${COLMAP_MODEL}
cmd="colmap image_registrator --database_path ${COLMAP_DIR}/database.db --input_path ${INITIAL_DIR}/colmap/sparse/0/ --output_path ${COLMAP_MODEL}"
echo $cmd
eval $cmd

echo "Step 5 - Re-run bundle adjustment"
cmd="colmap bundle_adjuster --input_path ${COLMAP_MODEL} --output_path ${COLMAP_MODEL} --BundleAdjustment.refine_principal_point 1"
echo $cmd
eval $cmd

echo "6. Alignment with Robot pose model - Requires a robot pose list retrieved from ROS (image_name X1 Y1 Z1)"
COLMAP_GEO=${COLMAP_DIR}/sparse_geo
mkdir $COLMAP_GEO
echo "colmap model_aligner --input_path ${COLMAP_MODEL} --output_path ${COLMAP_GEO} --alignment_max_error 1 --ref_is_gps 0 --ref_images_path ${NEW_POSE_FILE}"

echo "7. Convert the binary model to txt"
echo "colmap model_converter --input_path ${COLMAP_GEO} --output_path ${COLMAP_GEO} --output_type TXT"
