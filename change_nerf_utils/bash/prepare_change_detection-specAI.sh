#!/bin/bash

#This is for processing a recording from the spectacular AI app, aligning it with a prior recording,
# and generating the necessary depth images. A visualization is generated as a tool
# to demonstrate a successful alignment


#Check if CHANGE_HOME is set
if ! source is_home_set.sh; then
    echo "Failed to source is_home_set.sh" >&2
    exit 1
fi

BASH_HOME=$CHANGE_HOME/change_nerf_utils/bash
PYTHON_HOME=$CHANGE_HOME/change_nerf_utils/src/change_nerf_utils

# Recover the initial nerfstudio directory and the config directory
BASE_COLMAP_DIR="colmap/sparse/0"
BASE_CONFIG_DIR=$1
delimiter="outputs"
INITIAL_DIR="${BASE_CONFIG_DIR%$delimiter*}"
BASE_NERFSTUDIO_DIR="${INITIAL_DIR}/nerf_colmap"

# We are assuming that the recording was already unzipped - files will be stored in same directory as recording
RECORDING=$2
cd ${RECORDING}/..
SAVE_DIR=$(pwd)
echo "Save Dir = ${SAVE_DIR}"

#Check for a successfully extracted recording
TMP_NERF_DIR="${SAVE_DIR}/nerf_data"
if [[ ! -f $TMP_NERF_DIR/transforms.json ]]; then 
	echo "" 
	echo "STEP 1 - extract the frames using the sai-cli function to an arbitrary directory" 
	cmd="sai-cli process ${RECORDING} --key_frame_distance=0.05 ${TMP_NERF_DIR}"
	echo $cmd
	eval $cmd
	if [[ ! -f $TMP_NERF_DIR/transforms.json ]]; then 
		echo "sai-cli process failed - exiting"
		exit 1
	fi
fi

if [[ ! -d color ]];then
	echo "" 
	echo "STEP 2 - create the structure for change detection" 
	echo "Run from $(pwd)" 
	echo "cp -r ${TMP_NERF_DIR}/images color"
	cp -r ${TMP_NERF_DIR}/images color
fi

SPARSE_COMBINED="$SAVE_DIR/colmap_combined/sparse_combined/"  # directory where initial combined registration is stored
NEW_IMAGE_COUNT=$(grep new ${SPARSE_COMBINED}/0/images.txt | wc -l)
echo "Number of matched images=$NEW_IMAGE_COUNT"
if [ "$NEW_IMAGE_COUNT" -lt "50" ]; then
	echo "" 
	echo "STEP 3 - need to register the images with the nerf model" 
	cd ${BASH_HOME} 
	echo "Run from $(pwd)" 
	cmd="./register_new_images-nodepth.sh ${BASE_NERFSTUDIO_DIR} ${SAVE_DIR}"
	echo $cmd
	eval $cmd
	if [[ ! -f $SPARSE_COMBINED/0/images.txt ]];then 
		exit 1
	fi
else
    echo "New image count exceeds 50, no need for new registration"
fi

NEW_IMAGE_COUNT=$(grep new ${SPARSE_COMBINED}/0/images.txt | wc -l)
echo "Number of matched images=$NEW_IMAGE_COUNT"
if [ "$NEW_IMAGE_COUNT" -lt "50" ]; then
	echo "Not enough frames matched to the baseline room - exiting"
	exit 1
fi

SPARSE="${SAVE_DIR}/colmap_combined/sparse"
MERGED_COLMAP_DIR="${SAVE_DIR}/colmap_combined/sparse_merged" #output from model merger
if [[ ! -f $SPARSE/0/images.txt ]];then 
	echo "" 
	echo "STEP 4 - model_merger is good for matching the original frame of reference when creating new images. Model_aligner does not change the database"
	cd ${SAVE_DIR} 
	echo "Run from $(pwd)" 
	if [[ ! -f $MERGED_COLMAP_DIR/0/images.bin ]];then 
		echo "mkdir -p $MERGED_COLMAP_DIR/0/"
		mkdir -p $MERGED_COLMAP_DIR/0/
		cmd="colmap model_merger --input_path1 ${BASE_NERFSTUDIO_DIR}/${BASE_COLMAP_DIR} --input_path2 ${SPARSE_COMBINED}/0/ --output_path ${MERGED_COLMAP_DIR}/0/"
		echo $cmd
		eval $cmd
	fi

	rm $SPARSE
	echo "ln -s $MERGED_COLMAP_DIR $SPARSE" 
	ln -s $MERGED_COLMAP_DIR $SPARSE

	echo "" 
	echo "STEP 5 - create the new transforms and model txt files" 
	echo "Run from $(pwd)" 
	cmd="python ${PYTHON_HOME}/colmap_to_json.py ${MERGED_COLMAP_DIR}/0/ ${MERGED_COLMAP_DIR}/0/" 
	echo $cmd
	eval $cmd
	cmd="colmap model_converter --input_path ${MERGED_COLMAP_DIR}/0/ --output_path ${MERGED_COLMAP_DIR}/0/ --output_type TXT"
	echo $cmd
	eval $cmd	
fi

cp ${MERGED_COLMAP_DIR}/0/transforms.json $SAVE_DIR
cp ${MERGED_COLMAP_DIR}/0/sparse_pc.

LOCAL_RENDER_DIR=${SAVE_DIR}/renders
RENDER_IMAGE_COUNT=$( ls $LOCAL_RENDER_DIR/rgb*.png | wc -l )
echo "Number of rendered images=$RENDER_IMAGE_COUNT vs aligned new images=$NEW_IMAGE_COUNT"
if [ "$NEW_IMAGE_COUNT" -ne "$RENDER_IMAGE_COUNT" ]; then 
	echo "" 
	echo "STEP 6 - create all images" 
	cd ${INITIAL_DIR} 
	echo "Run from $(pwd)" 
	mkdir ${LOCAL_RENDER_DIR} 
	cmd="python ${PYTHON_HOME}/render_transform.py ${BASE_CONFIG_DIR} ${SAVE_DIR}/transforms.json ${LOCAL_RENDER_DIR} --image-type all --name-filter new_frame"
	echo $cmd
	eval $cmd
fi

#Just delete the existing geo directory - this step is fairly fast
GEO_COLMAP_DIR="${SAVE_DIR}/colmap_combined/sparse_geo"
echo ""
echo "STEP 7 - Create the geo mapped coordinate system for pointcloud creation"
echo "rm -rf $GEO_COLMAP_DIR"
rm -rf $GEO_COLMAP_DIR
echo "mkdir -p $GEO_COLMAP_DIR/0"
mkdir -p $GEO_COLMAP_DIR/0
cmd="colmap model_aligner --input_path $MERGED_COLMAP_DIR/0 --output_path $GEO_COLMAP_DIR/0 --alignment_max_error 0.3 --ref_is_gps 0 --ref_images_path $BASE_NERFSTUDIO_DIR/../camera_pose.txt"
echo $cmd
eval $cmd
cmd="python ${PYTHON_HOME}/colmap_to_json.py ${GEO_COLMAP_DIR}/0 ${GEO_COLMAP_DIR}/0" 
echo $cmd
eval $cmd
cmd="colmap model_converter --input_path ${GEO_COLMAP_DIR}/0 --output_path ${GEO_COLMAP_DIR}/0 --output_type TXT"
echo $cmd
eval $cmd

echo ""
echo "STEP 8 - visualize the results"
echo "Run from ${CHANGE_HOME}/change_pcloud_utils/"
echo "python change_pcloud_utils/visualize_colmap.py ${BASE_CONFIG_DIR} ${SAVE_DIR} --frame_keyword new_frame"


