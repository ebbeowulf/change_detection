#!/bin/bash

# Need to run roscore and register.launch first...

DARRAY=($(ls -d -- /data2/datasets/scannet/scans/*))

CLIP_TARGET=$1

cd ../scripts/llm_evaluation

for value in "${DARRAY[@]:100:150}"
do
    MAP_SUMMARY="$value/raw_output/save_results/${CLIP_TARGET}.summary.json"
    # SAVE_FILE="$value/raw_output/save_results/${CLIP_TARGET}.nvidia_llama.room.json"
    SAVE_FILE="$value/raw_output/save_results/${CLIP_TARGET}.nvidia_llama.object.json"
    cmd="python nvidia_llama.py $MAP_SUMMARY $CLIP_TARGET $SAVE_FILE"
    echo $cmd
    eval $cmd
done
