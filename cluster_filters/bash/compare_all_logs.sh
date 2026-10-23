#!/bin/bash
# Run compare_logs.py on every VLM log in one environment, adding one summary row per run/prompt.
#
# Usage:
#   ./compare_all.sh ENV_DIR PREDICTIONS OUTPUT_CSV IMAGE_THRESHOLD [CLUSTER_THRESHOLD] [VLM_THRESHOLD]
#   IMAGE_THRESHOLD:   DNN score at or above which an image counts as changed
#   CLUSTER_THRESHOLD: DNN: fraction of changed images at or above which a cluster is changed (default 0.5)
#   VLM_THRESHOLD:     VLM: fraction of new-object views at or above which a cluster is changed (default 0.5)
# Example:
#   ./compare_all.sh /data3/datasets/smart_change/j234 \
#       /data2/datasets/change_dataset/runs/siamese_j234/predictions.csv \
#       /data2/datasets/change_dataset/runs/vlm_vs_dnn_j234.csv 0.1

ENV_DIR=${1%/}
PREDICTIONS=$2
OUTPUT_CSV=$3
IMAGE_THRESHOLD=$4
CLUSTER_THRESHOLD=${5:-0.5}
VLM_THRESHOLD=${6:-0.5}

if [ -z "$IMAGE_THRESHOLD" ]; then
    echo "usage: $0 ENV_DIR PREDICTIONS OUTPUT_CSV IMAGE_THRESHOLD [CLUSTER_THRESHOLD] [VLM_THRESHOLD]"
    exit 1
fi

cd ${CHANGE_HOME}/resnet_filter/src/resnet_filter

# which VLM logs to compare (edit to switch label source or VLM model)
RESULTS_SUBDIR=save_results_unfiltered/sam3_0.1_openVocab
VLM_FILES="before_unannotated_and_after_filter-*-qwen2.5vl:32b.json"

ENV_NAME=$(basename "$ENV_DIR")

find -L "$ENV_DIR" -path "*/$RESULTS_SUBDIR/$VLM_FILES" | sort | while read -r VLM_LOG; do
    # .../j234/T1/changes/changes6/save_results_unfiltered/... -> run j234/T1__changes__changes6
    RUN_DIR=${VLM_LOG%/$RESULTS_SUBDIR/*}
    RUN=$ENV_NAME/$(echo "${RUN_DIR#$ENV_DIR/}" | sed 's#/#__#g')
    # before_unannotated_and_after_filter-electronics-qwen2.5vl:32b.json -> electronics
    PROMPT=$(basename "$VLM_LOG" | cut -d- -f2)

    echo "=== $RUN  $PROMPT"
    python compare_logs.py "$VLM_LOG" "$PREDICTIONS" "$RUN" "$PROMPT" --image_threshold "$IMAGE_THRESHOLD" --cluster_threshold "$CLUSTER_THRESHOLD" --vlm_threshold "$VLM_THRESHOLD" \
        --output_csv "$OUTPUT_CSV" \
        || echo "    SKIPPED"
done