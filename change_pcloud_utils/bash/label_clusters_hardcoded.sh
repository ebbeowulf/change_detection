#!/bin/bash

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3

#CHANGE_DIR=( '/data2/datasets/south_haven/specAI2-baseline_sparse/' '/data2/datasets/living_room/specAI2/' )
# CHANGE_DIR=( '/data2/datasets/living_room/specAI2/' )
CHANGE_DIR=( '/data3/datasets/smart_change/j234/T1/changes/' '/data3/datasets/smart_change/j234/T2/changes/' '/data3/datasets/smart_change/j234/T3/changes/' )
# CHANGE_DIR=( '/data3/datasets/smart_change/s120/T1/changes/' '/data3/datasets/smart_change/s120/T2/changes/' '/data3/datasets/smart_change/s120/T3/changes/' )
CHANGE_DIR=( '/data3/datasets/smart_change/j234/T1/changes/' '/data3/datasets/smart_change/j234/T2/changes/' '/data3/datasets/smart_change/j234/T3/changes/' '/data3/datasets/smart_change/s120/T1/changes/' '/data3/datasets/smart_change/s120/T2/changes/' '/data3/datasets/smart_change/s120/T3/changes/')
#CHANGE_DIR=( '/data2/datasets/south_haven/specAI2-baseline_sparse/' '/data2/datasets/living_room/specAI2/' '/data3/datasets/smart_change/j234/T1/changes/' '/data3/datasets/smart_change/j234/T2/changes/' '/data3/datasets/smart_change/j234/T3/changes/' )
# CHANGE_DIR=( '/data3/datasets/smart_change/j234/T1/changes/' )
#RESULTS_DIR=( 'save_results_unfiltered/sam3_0.1_change' 'save_results_unfiltered/sam3_0.1_openVocab' 'save_results_unfiltered/clipseg_0.1_change' 'save_results_unfiltered/clipseg_0.1_openVocab' 'save_results_noBlur/sam3_0.1_change' 'save_results_noBlur/clipseg_0.1_change' )
# RESULTS_DIR=( 'save_results_unfiltered/sam3_0.1_change' 'save_results_unfiltered/sam3_0.1_openVocab' 'save_results_unfiltered/clipseg_0.1_change' 'save_results_unfiltered/clipseg_0.1_openVocab' )
# RESULTS_DIR=( 'save_results_unfiltered/sam3_0.1_change' 'save_results_unfiltered/sam3_0.1_openVocab' )
RESULTS_DIR=( 'save_results_unfiltered/sam3_0.1_openVocab')

#QUERIES="'clothing' 'dishes' 'general clutter' 'small items'"
QUERIES="'electronics' 'trash' 'general clutter' 'small items'"
# QUERIES="'small items'"
#QUERIES="'clothing' 'dishes' 'electronics' 'trash' 'general clutter' 'small items'"
#FILTERS="pct_valid_filter is_pickup_filter combo_pctV_pickup"
# FILTERS="is_pickup_filter"
# FILTERS="before_and_after_filter"
FILTERS="before_unannotated_and_after_filter"

cd ${CHANGE_HOME}/change_pcloud_utils/src/change_pcloud_utils
for resD in "${RESULTS_DIR[@]}"
do
    tgt_dir=""
    for cDir in "${CHANGE_DIR[@]}"
    do
        mapfile -t SUBDIR < <(find "$cDir" -mindepth 1 -maxdepth 1 -type d)
        for subD in "${SUBDIR[@]}"
        do 
            tgt_dir="${tgt_dir}${subD}/${resD} "
        done
    done

    #cmd="python label_clusters.py --tgt_dir $tgt_dir --queries $QUERIES --filters $FILTERS"
    cmd="python label_clusters.py --tgt_dir $tgt_dir --queries $QUERIES --filters $FILTERS --label_suffix labels.tight.json"
    echo $cmd
    echo " "
    echo " "
    eval $cmd
done
