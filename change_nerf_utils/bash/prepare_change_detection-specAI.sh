#!/bin/bash

#This is for processing a recording from the spectacular AI app, aligning it with a prior recording,
# and generating the necessary depth images. A visualization is generated as a tool
# to demonstrate a successful alignment
CHANGE_HOME=/home/emartinso/ros_ws/src/research/change_detection
BASH_HOME=${CHANGE_HOME}/bash
PYTHON_HOME=${CHANGE_HOME}/scripts/change_detection
PCLOUD_PYTHON_HOME=/home/emartinso/ros_ws/src/research/pcloud_models/scripts/pcloud_models
#BASE_NERFSTUDIO_DIR="/home/emartinso/data/living_room/INITIAL/specAI_morning/nerf_specAI2colmap_v2/"
BASE_COLMAP_DIR="colmap/sparse/0"
#CONFIG_DIR="outputs/splatfacto/2025-06-26_175947/"

# Recover the initial nerfstudio directory and the config directory
BASE_CONFIG_DIR=$1
delimiter="outputs"
BASE_NERFSTUDIO_DIR="${BASE_CONFIG_DIR%$delimiter*}"
CONFIG_DIR="${BASE_CONFIG_DIR#*$BASE_NERFSTUDIO_DIR}"

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
	echo "sai-cli process ${RECORDING} --key_frame_distance=0.05 ${TMP_NERF_DIR}"
	sai-cli process ${RECORDING} --key_frame_distance=0.05 ${TMP_NERF_DIR}
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

NEW_IMAGE_COUNT=$(grep new $SAVE_DIR/colmap_combined/sparse_combined/images.txt | wc -l)
echo "Number of matched images=$NEW_IMAGE_COUNT"
if [ "$NEW_IMAGE_COUNT" -lt "50" ]; then
	echo "" 
	echo "STEP 3 - need to register the images with the nerf model" 
	cd ${BASH_HOME} 
	echo "Run from $(pwd)" 
	echo "./register_new_images-nodepth.sh ${BASE_NERFSTUDIO_DIR} ${SAVE_DIR}"
	./register_new_images-nodepth.sh ${BASE_NERFSTUDIO_DIR} ${SAVE_DIR}
        if [[ ! -f $SAVE_DIR/colmap_combined/sparse_combined/images.txt ]];then 
		exit 1
	fi
fi

NEW_IMAGE_COUNT=$(grep new $SAVE_DIR/colmap_combined/sparse_combined/images.txt | wc -l)
echo "Number of matched images=$NEW_IMAGE_COUNT"
if [ "$NEW_IMAGE_COUNT" -lt "50" ]; then
	echo "Not enough frames matched to the baseline room - exiting"
	exit 1
fi

LOCAL_COLMAP_DIR="${SAVE_DIR}/colmap_combined/sparse"
MERGED_COLMAP_DIR="${SAVE_DIR}/colmap_combined/sparse_merged"
if [[ ! -f $LOCAL_COLMAP_DIR/images.txt ]];then 
	echo "" 
	echo "STEP 4 - model_merger is good for matching the original frame of reference when creating new images. Model_aligner does not change the database"
	cd ${SAVE_DIR} 
	echo "Run from $(pwd)" 
        if [[ ! -f $MERGED_COLMAP_DIR/images.bin ]];then 
		echo "mkdir -p $MERGED_COLMAP_DIR"
		mkdir -p $MERGED_COLMAP_DIR
	        echo "colmap model_merger --input_path1 ${BASE_NERFSTUDIO_DIR}/${BASE_COLMAP_DIR} --input_path2 ${SAVE_DIR}/colmap_combined/sparse_combined --output_path ${MERGED_COLMAP_DIR}"
	        colmap model_merger --input_path1 ${BASE_NERFSTUDIO_DIR}/${BASE_COLMAP_DIR} --input_path2 ${SAVE_DIR}/colmap_combined/sparse_combined --output_path ${MERGED_COLMAP_DIR}
	fi

	echo "ln -s $MERGED_COLMAP_DIR ${LOCAL_COLMAP_DIR}" 
	ln -s $MERGED_COLMAP_DIR ${LOCAL_COLMAP_DIR}

	echo "" 
	echo "STEP 5 - create the new transforms and model txt files" 
	echo "Run from $(pwd)" 
	echo "python ${PYTHON_HOME}/colmap_to_json.py ${LOCAL_COLMAP_DIR} ${LOCAL_COLMAP_DIR}" 
	python ${PYTHON_HOME}/colmap_to_json.py ${LOCAL_COLMAP_DIR} ${LOCAL_COLMAP_DIR}
	echo "colmap model_converter --input_path ${LOCAL_COLMAP_DIR} --output_path ${LOCAL_COLMAP_DIR} --output_type TXT"
	colmap model_converter --input_path ${LOCAL_COLMAP_DIR} --output_path ${LOCAL_COLMAP_DIR} --output_type TXT

fi

LOCAL_RENDER_DIR=${SAVE_DIR}/renders
RENDER_IMAGE_COUNT=$( ls $LOCAL_RENDER_DIR/rgb*.png | wc -l )
echo "Number of rendered images=$RENDER_IMAGE_COUNT vs aligned new images=$NEW_IMAGE_COUNT"
if [ "$NEW_IMAGE_COUNT" -ne "$RENDER_IMAGE_COUNT" ]; then 
	echo "" 
	echo "STEP 6 - create all images" 
	cd ${BASE_NERFSTUDIO_DIR} 
	echo "Run from $(pwd)" 
	mkdir ${LOCAL_RENDER_DIR} 
	cd ${BASE_NERFSTUDIO_DIR} 
	echo "python ${PYTHON_HOME}/render_transform.py ${CONFIG_DIR} ${LOCAL_COLMAP_DIR}/transforms.json ${LOCAL_RENDER_DIR} --image-type all --name-filter new_frame"
	python ${PYTHON_HOME}/render_transform.py ${CONFIG_DIR} ${LOCAL_COLMAP_DIR}/transforms.json ${LOCAL_RENDER_DIR} --image-type all --name-filter new_frame
fi

#Just delete the existing geo directory - this step is fairly fast
GEO_COLMAP_DIR="${SAVE_DIR}/colmap_combined/sparse_geo"
echo ""
echo "STEP 7 - Create the geo mapped coordinate system for pointcloud creation"
echo "rm -rf $GEO_COLMAP_DIR"
rm -rf $GEO_COLMAP_DIR
echo "mkdir -p $GEO_COLMAP_DIR"
mkdir -p $GEO_COLMAP_DIR
echo "colmap model_aligner --input_path $MERGED_COLMAP_DIR --output_path $GEO_COLMAP_DIR --alignment_max_error 0.3 --ref_is_gps 0 --ref_images_path $BASE_NERFSTUDIO_DIR/../camera_pose.txt"
colmap model_aligner --input_path $MERGED_COLMAP_DIR --output_path $GEO_COLMAP_DIR --alignment_max_error 0.3 --ref_is_gps 0 --ref_images_path $BASE_NERFSTUDIO_DIR/../camera_pose.txt
echo "python ${PYTHON_HOME}/colmap_to_json.py ${GEO_COLMAP_DIR} ${GEO_COLMAP_DIR}" 
python ${PYTHON_HOME}/colmap_to_json.py ${GEO_COLMAP_DIR} ${GEO_COLMAP_DIR}
echo "colmap model_converter --input_path ${GEO_COLMAP_DIR} --output_path ${GEO_COLMAP_DIR} --output_type TXT"
colmap model_converter --input_path ${GEO_COLMAP_DIR} --output_path ${GEO_COLMAP_DIR} --output_type TXT

echo ""
echo "STEP 8 - visualize the results"
echo "Run from $(pwd)"
echo "python ${PCLOUD_PYTHON_HOME}/visualize_colmap.py ${BASE_NERFSTUDIO_DIR}/${CONFIG_DIR} ${SAVE_DIR} --frame_keyword new_frame"


