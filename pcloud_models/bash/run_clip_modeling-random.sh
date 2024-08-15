#!/bin/bash

# Need to run roscore and register.launch first...

DARRAY=($(ls -d -- /data2/datasets/scannet/scans/*))
DARRAY=( $(shuf -e "${DARRAY[@]:100:150}") )

CLIP_TARGET=$1
THRESHOLD=$2

cd ../scripts/pcloud_models

for value in "${DARRAY[@]}"
do
    cmd="python scannet_processing.py $value --targets $CLIP_TARGET --threshold $THRESHOLD"
    echo $cmd
    eval $cmd

    cmd="python determine_relationship.py $value $CLIP_TARGET"
    echo $cmd
    eval $cmd
done
