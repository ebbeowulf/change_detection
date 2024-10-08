#!/bin/bash

# Need to run roscore and register.launch first...

DARRAY=($(ls -d -- /data3/datasets/scannet/scans/scene*))

TARGET=$1
THRESHOLD=$2

SCRIPTS_DIR="$(pwd)/../scripts/pcloud_models"
cd $SCRIPTS_DIR

#for value in "${DARRAY[@]:100:150}"
for value in "${DARRAY[@]:0:250}"
do
    echo "Checking $value"
    # We need to extract the scannet labels with segmentation results
    LABEL_FLT_DIR="$value/label-filt/"
    if  [ ! -d $LABEL_FLT_DIR ];then
        cd $value
        cmd="unzip *label-filt.zip"
        echo $cmd
        eval $cmd
        cd $SCRIPTS_DIR
    else
	    echo "found label filt dir"
    fi
   
    # We need to create the labeled point clouds
    LABEL_FILE="$value/raw_output/save_results/all_labels.json"
    LABEL_PLY="$value/raw_output/save_results/$TARGET.labeled.ply"
    if [ ! -f $LABEL_FILE ] || [ ! -f "$LABEL_PLY" ]; then
        cmd="python object_pcloud_from_scannet.py $value '$TARGET'"
        echo $cmd
        eval $cmd
    else
	echo "found labeled point clouds"
    fi

    # We need to create the master pointcloud
    COMBO_PCL_FILE="$value/raw_output/save_results/combined.ply"
    if [ ! -f $COMBO_PCL_FILE ]; then
        cmd="python visualize_scannet.py $value --headless"
        echo $cmd
        # eval $cmd
    else
	echo "found master point cloud"
    fi

    # We want to create the CLIP segmentation - not strictly necessary
    #   for labeling, but important in the long run regardless
    CLIP_PCL_FILE="$value/raw_output/save_results/$TARGET.raw.pkl"
    if [ ! -f "$CLIP_PCL_FILE" ]; then
        cmd="python scannet_processing.py $value --targets $TARGET --threshold $THRESHOLD"
        echo $cmd
        eval $cmd
    else
	    echo "found raw pkl file: $CLIP_PCL_FILE"
    fi
done
