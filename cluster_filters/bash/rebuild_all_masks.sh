#!/bin/bash

MODEL=$1

# SPECAI_DIR=(
#     '/data2/datasets/living_room/specAI2/6_24_v1/'
#     '/data2/datasets/living_room/specAI2/6_24_v2/'  
#     '/data2/datasets/living_room/specAI2/6_25_v1/'  
#     '/data2/datasets/living_room/specAI2/6_25_v2/'
#     '/data2/datasets/south_haven/specAI2-baseline_sparse/7_24_v1/'
#     '/data2/datasets/south_haven/specAI2-baseline_sparse/7_24_v2/'  
#     '/data2/datasets/south_haven/specAI2-baseline_sparse/7_25_v1/'  
#     '/data2/datasets/south_haven/specAI2-baseline_sparse/7_25_v2/'
# )

# SPECAI_PROMPTS='"clothing" "dishes" "general clutter" "small items"'
# cd ${CHANGE_HOME}/resnet_filter/src/resnet_filter
# for resD in "${SPECAI_DIR[@]}"
# do
#     cmd="python build_delta_change_masks.py $resD ${SPECAI_PROMPTS} --model $MODEL"
#     echo $cmd
#     eval $cmd
# done

ROBOT_DIR=(
    '/data2/datasets/j234/T1/changes/changes5/'    
    '/data2/datasets/j234/T3/changes2/changes8/'
    '/data2/datasets/j234/T1/changes/changes6/'  
    '/data2/datasets/j234/T3/changes2/changes9/'
    '/data2/datasets/j234/T2/changes/changes10/'   
    '/data2/datasets/j234/T3/changes/changes1/'
    '/data2/datasets/j234/T2/changes/changes7/'    
    '/data2/datasets/j234/T3/changes/changes2/'
    '/data2/datasets/j234/T2/changes/changes9/'   
    '/data2/datasets/j234/T3/changes/changes3/'
    '/data2/datasets/j234/T3/changes2/changes10/'  
    '/data2/datasets/j234/T3/changes/changes4/'
    '/data2/datasets/j234/T3/changes2/changes6/'   
    '/data2/datasets/j234/T3/changes/changes5/'
    '/data2/datasets/j234/T3/changes2/changes7/'
    '/data2/datasets/j234/T1/changes/changes5/'
    '/data2/datasets/j234/T1/changes/changes6/'
    '/data2/datasets/j234/T2/changes8/'
    '/data2/datasets/j234/T2/changes/changes10/'
    '/data2/datasets/j234/T2/changes/changes7/'
    '/data2/datasets/j234/T2/changes/changes9/'
    '/data2/datasets/j234/T3/changes2/changes10/'
    '/data2/datasets/j234/T3/changes2/changes6/'
    '/data2/datasets/j234/T3/changes2/changes7/'
    '/data2/datasets/j234/T3/changes2/changes8/'
    '/data2/datasets/j234/T3/changes2/changes9/'
    '/data2/datasets/j234/T3/changes/changes1/'
    '/data2/datasets/j234/T3/changes/changes2/'
    '/data2/datasets/j234/T3/changes/changes3/'
    '/data2/datasets/j234/T3/changes/changes4/'
    '/data2/datasets/j234/T3/changes/changes5/'
    '/data2/datasets/s120/T1/changes/changes1/'  
    '/data2/datasets/s120/T3/changes/changes1/'
    '/data2/datasets/s120/T1/changes/changes2/'  
    '/data2/datasets/s120/T3/changes/changes2/'
    '/data2/datasets/s120/T2/changes/change1/'
)

ROBOT_PROMPTS='"electronics" "trash" "general clutter" "small items"'
cd ${CHANGE_HOME}/resnet_filter/src/resnet_filter
for resD in "${ROBOT_DIR[@]}"
do
    cmd="python build_delta_change_masks.py $resD ${ROBOT_PROMPTS} --model $MODEL"
    echo $cmd
    eval $cmd
done