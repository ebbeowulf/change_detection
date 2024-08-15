#!/bin/bash

# Need to run roscore and register.launch first...

DARRAY=($(ls -d -- /data2/datasets/scannet/scans/*))

CLIP_TARGET=$1
THRESHOLD=$2

cd ../scripts/pcloud_models

for value in "${DARRAY[@]:100:150}"
do
    cmd="python scannet_processing.py $value --targets $CLIP_TARGET --threshold $THRESHOLD"
    echo $cmd
    eval $cmd

    cmd="python determine_relationship.py $value $CLIP_TARGET"
    echo $cmd
    eval $cmd
done
