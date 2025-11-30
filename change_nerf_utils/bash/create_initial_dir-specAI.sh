#!/bin/bash

#Check if CHANGE_HOME is set
if ! source is_home_set.sh; then
    echo "Failed to source is_home_set.sh" >&2
    exit 1
fi

BASH_HOME=$CHANGE_HOME/change_nerf_utils/bash
PYTHON_HOME=$CHANGE_HOME/change_nerf_utils/src/change_nerf_utils

RECORDING=$1                                                                                          
cd $RECORDING/..
BASE_DIR=$(pwd) 

# Process the recording...
SAI_NERF_DIR=$BASE_DIR/sai_nerf_data
echo $SAI_NERF_DIR
if [[ ! -f $SAI_NERF_DIR/transforms.json ]]; then
	echo ""
       	echo "STEP 1 - extract the sai data"
	echo "sai-cli process $RECORDING --key_frame_distance=0.05 $SAI_NERF_DIR"
	sai-cli process $RECORDING --key_frame_distance=0.05 $SAI_NERF_DIR
fi
COLOR_IMAGE_DIR=${SAI_NERF_DIR}/images

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


# Step 1: Build the camera_pose.txt file - don't worry about doing this musltiple times
echo "python $PYTHON_HOME/convert_images_txt_to_pose.py $SAI_NERF_DIR/colmap/sparse/0/images.txt > camera_pose.txt"
python $PYTHON_HOME/convert_images_txt_to_pose.py $SAI_NERF_DIR/colmap/sparse/0/images.txt > camera_pose.txt

# Reprocess the directory using colmap
# COLMAP_NERF_DIR=$BASE_DIR/nerf_colmap
# if [[ ! -f $COLMAP_NERF_DIR/transforms.json ]]; then
# 	echo "mkdir -p $COLMAP_NERF_DIR"
# 	mkdir -p $COLMAP_NERF_DIR
# 	echo "cp -r $SAI_NERF_DIR/images $COLMAP_NERF_DIR"
# 	cp -r $SAI_NERF_DIR/images $COLMAP_NERF_DIR
# 	echo "ns-process-data images --data $COLMAP_NERF_DIR/images/ --output-dir $COLMAP_NERF_DIR/ --skip-image-processing"
# 	ns-process-data images --data $COLMAP_NERF_DIR/images/ --output-dir $COLMAP_NERF_DIR/ --skip-image-processing
# fi
COLMAP_NERF_DIR=$BASE_DIR/nerf_colmap
cd $BASE_DIR/
SPARSE=$COLMAP_NERF_DIR/colmap/sparse_orig
mkdir -p $SPARSE
ln -s $COLOR_IMAGE_DIR $COLMAP_NERF_DIR/images
ln -s $SPARSE $COLMAP_NERF_DIR/colmap/sparse # temporary link for processing - will be removed later

if [[ ! -f $SPARSE/0/images.bin ]];then
    # Alternative method that loads data via colmap without the ns-process-data step
    COLMAP_DB=$COLMAP_NERF_DIR/colmap/database.db

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

if [[ ! -f $SPARSE/0/images.txt ]];then
	cmd="colmap model_converter --input_path $SPARSE/0/ --output_path $SPARSE/0/ --output_type TXT"
	echo $cmd
	eval $cmd
fi

# Identify the sparse directory with the best coverage of the initial poses
#   note that we only check the 0 + 1 directories. If >1 exists, then it won't be used
./set_best_colmap_subdir.sh $SPARSE jpg

# Go ahead and do the model alignment...
SPARSE_GEO=$COLMAP_NERF_DIR/colmap/sparse_geo
if [[ ! -f $SPARSE_GEO/0/images.txt ]];then
	mkdir -p $SPARSE_GEO/0
	cmd="colmap model_aligner --input_path $SPARSE/0 --output_path $SPARSE_GEO/0 --alignment_max_error 0.3 --ref_is_gps 0 --ref_images_path $BASE_DIR/camera_pose.txt"
	echo $cmd
	eval $cmd

	cmd="colmap model_converter --input_path $SPARSE_GEO/0/ --output_path $SPARSE_GEO/0/ --output_type TXT"
	echo $cmd
	eval $cmd

    cmd="python $PYTHON_HOME/colmap_to_json.py $SPARSE_GEO/0 $SPARSE_GEO/0/"
    echo $cmd
    eval $cmd

    # Convert the aligned COLMAP model to transforms.json and sparse_pc.ply - copy these to the nerf_colmap directory if
    #   you want to use them during training or with depth images
    rm -rf $COLMAP_NERF_DIR/colmap/sparse
    ln -s $SPARSE_GEO $COLMAP_NERF_DIR/colmap/sparse
    cp $SPARSE_GEO/0/transforms.json $COLMAP_NERF_DIR/transforms.json
    cp $SPARSE_GEO/0/sparse_pc.ply $COLMAP_NERF_DIR/sparse_pc.ply

fi

# Last step - run nerfstudio
cd $COLMAP_NERF_DIR
echo "Run the following command from the $BASE_DIR to start training:"
echo "ns-train splatfacto --data nerf_colmap"
#ns-train splatfacto --data .
