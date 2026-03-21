#!/bin/bash

# Activate Conda and ROS 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3

# CONFIG_DIR='/data2/datasets/south_haven/INITIAL/baseline_sparse/outputs/nerf_colmap/splatfacto/2025-11-19_201443'
# CHANGES_DIR='/data2/datasets/south_haven/specAI2-baseline_sparse/'
# RESULTS_DIR=( 'save_results_unfiltered/sam3_0.1_change' 'save_results_unfiltered/sam3_0.1_openVocab' 'save_results_unfiltered/clipseg_0.1_change' 'save_results_unfiltered/clipseg_0.1_openVocab' 'save_results_noBlur/sam3_0.1_change' 'save_results_noBlur/clipseg_0.1_change' )
# RESULTS_DIR=( 'save_results_unfiltered/sam3_0.1_change' 'save_results_unfiltered/sam3_0.1_openVocab' 'save_results_unfiltered/clipseg_0.1_change' 'save_results_unfiltered/clipseg_0.1_openVocab' )
RESULTS_DIR=( 'save_results_unfiltered/sam3_0.1_openVocab' )

CONFIG_DIR=$1
CHANGES_DIR=$2
QUERIES=$3

mapfile -t SUBDIR < <(find "$CHANGES_DIR" -mindepth 1 -maxdepth 1 -type d)

cd ${CHANGE_HOME}/change_pcloud_utils/src/change_pcloud_utils
# QUERIES="\"clothing\" \"dishes\" \"general clutter\" \"small items\""

FILTERS="before_and_after_filter"

for subD in "${SUBDIR[@]}"
do 
    # echo $subD
    for resD in "${RESULTS_DIR[@]}"
    do
        tgt_dir="$ROOT_DIR/$subD/$resD"
        if [[ -d $tgt_dir ]];then
            # cmd="python filter_clusters.py $tgt_dir --queries $QUERIES --filters $FILTERS --nerfacto_dir $CONFIG_DIR --frame_keyword color"
            cmd="python before_and_after_filter.py $tgt_dir $CONFIG_DIR --queries $QUERIES --frame_keyword color --filterType before_unannotated_and_after_filter"
            echo $cmd
            eval $cmd
        fi
    done
done
