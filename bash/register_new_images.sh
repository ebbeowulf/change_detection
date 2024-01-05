#!/bin/bash

# Before running this, need to run ns-process-data to build the default colmap pipeline

INITIAL_DIR=$1
NEW_DIR_ROOT=$2


COLMAP_DIR=${NEW_DIR_ROOT}/colmap_combined
IMAGE_DIR=${NEW_DIR_ROOT}/images_combined
VOCAB_TREE="/data2/datasets/office/vocab_tree_flickr100K_words256K.bin"

echo "Step 1 - Preparing directory"
echo "./prepare_dir.sh $INITIAL_DIR $NEW_DIR_ROOT"
./prepare_dir.sh $INITIAL_DIR $NEW_DIR_ROOT

echo "Step 2 - Extract features"
NEW_IMAGES=$NEW_DIR_ROOT/new_images.txt
echo "colmap feature_extractor --database_path ${COLMAP_DIR}/database.db --image_path ${IMAGE_DIR} --image_list_path ${NEW_IMAGES} --ImageReader.single_camera 1 --ImageReader.existing_camera_id 1 --SiftExtraction.use_gpu 1"
colmap feature_extractor --database_path ${COLMAP_DIR}/database.db --image_path ${IMAGE_DIR} --image_list_path ${NEW_IMAGES} --ImageReader.single_camera 1 --ImageReader.existing_camera_id 1 --SiftExtraction.use_gpu 1

echo "Step 3 - Match features with existing images"
echo "colmap vocab_tree_matcher --database_path ${COLMAP_DIR}/database.db --VocabTreeMatching.vocab_tree_path $VOCAB_TREE --SiftMatching.use_gpu 1"
colmap vocab_tree_matcher --database_path ${COLMAP_DIR}/database.db --VocabTreeMatching.vocab_tree_path $VOCAB_TREE --SiftMatching.use_gpu 1

echo "Step 4 - Register images between old and new"
COLMAP_MODEL=${COLMAP_DIR}/sparse_combined
mkdir ${COLMAP_MODEL}
echo "colmap image_registrator --database_path ${COLMAP_DIR}/database.db --input_path ${INITIAL_DIR}/colmap/sparse/0/ --output_path ${COLMAP_MODEL}"
colmap image_registrator --database_path ${COLMAP_DIR}/database.db --input_path ${INITIAL_DIR}/colmap/sparse/0/ --output_path ${COLMAP_MODEL}

echo "Step 5 - Re-run bundle adjustment"
echo "colmap bundle_adjuster --input_path ${COLMAP_MODEL} --output_path ${COLMAP_MODEL} --BundleAdjustment.refine_principal_point 1"
colmap bundle_adjuster --input_path ${COLMAP_MODEL} --output_path ${COLMAP_MODEL} --BundleAdjustment.refine_principal_point 1

echo "6. Alignment with Robot pose model - Requires a robot pose list retrieved from ROS (image_name X1 Y1 Z1)"
COLMAP_GEO=${COLMAP_DIR}/sparse_geo
mkdir $COLMAP_GEO
echo "colmap model_aligner --input_path ${COLMAP_MODEL} --output_path ${COLMAP_GEO} --alignment_max_error 0.5 --ref_is_gps 0 --ref_images_path ${INITIAL_DIR}/pose.txt"

echo "7. Convert the binary model to txt"
echo "colmap model_converter --input_path ${COLMAP_GEO} --output_path ${COLMAP_GEO} --output_type TXT"
