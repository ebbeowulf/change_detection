#!/bin/bash

# Need to run roscore and register.launch first...

DARRAY=($(ls -d -- /data2/datasets/scannet/scans/scene*))
#DARRAY=($(ls -d -- /data3/datasets/scannet/scans/scene*))

TARGET1=$1

SCRIPTS_DIR="$(pwd)/../scripts/pcloud_models"
cd $SCRIPTS_DIR

# for value in "${DARRAY[@]:100:150}"
for value in "${DARRAY[@]:0:250}"
do
    # LABEL_FLT_DIR="$value/label-filt/"
    # if  [ ! -d $LABEL_FLT_DIR ];then
    #     cd $value
    #     cmd="unzip *label-filt.zip"
    #     echo $cmd
    #     eval $cmd
    #     cd $SCRIPTS_DIR
    # fi
   
    # LABEL_FILE="$value/raw_output/save_results/all_labels.json"
    # LABEL_PLY="$value/raw_output/save_results/$TARGET1.labeled.ply"
    # if [ ! -f $LABEL_FILE ] || [ ! -f $LABEL_PLY ]; then
    #     cmd="python object_pcloud_from_scannet.py $value $TARGET1"
    #     echo $cmd
    #     eval $cmd
    # fi

    ANNOT_FILE="$value/raw_output/save_results/annotations.json"
    #if [ ! -f $ANNOT_FILE ]; then
    if grep -q $TARGET1 $ANNOT_FILE; then
        echo "$TARGET1 already labeled for $value"
    else
        cmd="python labelling.py $value --targets \"$TARGET1\" "
        echo $cmd
        eval $cmd
    fi
done
