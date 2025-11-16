#!/bin/bash

CHANGE_HOME=/home/emartinso/ros_ws/src/research/change_detection
BASH_HOME=$CHANGE_HOME/bash
PYTHON_HOME=$CHANGE_HOME/scripts/change_detection                                                                                                                                                         
RECORDING=$1                                                                                          
cd $RECORDING/..
BASE_DIR=$(pwd) # Process the recording...
SAI_NERF_DIR=$BASE_DIR/sai_nerf_data
echo $SAI_NERF_DIR
if [[ ! -f $SAI_NERF_DIR/transforms.json ]]; then
	echo ""
       	echo "STEP 1 - extract the sai data"
	echo "sai-cli process $RECORDING --key_frame_distance=0.05 $SAI_NERF_DIR"
	sai-cli process $RECORDING --key_frame_distance=0.05 $SAI_NERF_DIR
fi

# Build the camera_pose.txt file - don't worry about doing this multiple times
echo "python $PYTHON_HOME/convert_images_txt_to_pose.py $SAI_NERF_DIR/colmap/sparse/0/images.txt > camera_pose.txt"
python $PYTHON_HOME/convert_images_txt_to_pose.py $SAI_NERF_DIR/colmap/sparse/0/images.txt > camera_pose.txt

# Reprocess the directory using colmap
COLMAP_NERF_DIR=$BASE_DIR/nerf_colmap
if [[ ! -f $COLMAP_NERF_DIR/transforms.json ]]; then
	echo "mkdir -p $COLMAP_NERF_DIR"
	mkdir -p $COLMAP_NERF_DIR
	echo "cp -r $SAI_NERF_DIR/images $COLMAP_NERF_DIR"
	cp -r $SAI_NERF_DIR/images $COLMAP_NERF_DIR
	echo "ns-process-data images --data $COLMAP_NERF_DIR/images/ --output-dir $COLMAP_NERF_DIR/ --skip-image-processing"
	ns-process-data images --data $COLMAP_NERF_DIR/images/ --output-dir $COLMAP_NERF_DIR/ --skip-image-processing
fi

SPARSE=$COLMAP_NERF_DIR/colmap/sparse
if [[ ! -f $SPARSE/0/images.txt ]];then
	echo "colmap model_converter --input_path $SPARSE/0/ --output_path $SPARSE/0/ --output_type TXT"
	colmap model_converter --input_path $SPARSE/0/ --output_path $SPARSE/0/ --output_type TXT
fi

# Go ahead and do the model alignment...
SPARSE_GEO=$COLMAP_NERF_DIR/colmap/sparse_geo
if [[ ! -f $SPARSE_GEO/0/images.txt ]];then
	mkdir -p $SPARSE_GEO/0
        echo "colmap model_aligner --input_path $SPARSE/0 --output_path $SPARSE_GEO/0 --alignment_max_error 0.3 --ref_is_gps 0 --ref_images_path $BASE_DIR/camera_pose.txt"
        colmap model_aligner --input_path $SPARSE/0 --output_path $SPARSE_GEO/0 --alignment_max_error 0.3 --ref_is_gps 0 --ref_images_path $BASE_DIR/camera_pose.txt  
	colmap model_converter --input_path $SPARSE_GEO/0/ --output_path $SPARSE_GEO/0/ --output_type TXT
fi

# Last step - run nerfstudio
cd $COLMAP_NERF_DIR
echo "ns-train splatfacto --data ."
ns-train splatfacto --data .
