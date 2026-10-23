#!/bin/bash
################################################################################
# run_filter_sequence.sh
#
# Run any sequence of filters on every change run under a root folder, then score
# the result against that run's labels. Replaces one script per combination.
#
# Usage:
#     ./run_filter_sequence.sh ROOT_DIR STAGES [FILTER_ARGS] [CONFIG_NAME] [OUTPUT_CSV]
#
#     ROOT_DIR     folder to search: one environment or several
#     STAGES       filters to run, in order, as ONE quoted string, e.g. 'siamese vlm_change'
#     FILTER_ARGS  options for those filters, as ONE quoted string (default: none), e.g.
#                  '--siamese_model /data2/datasets/change_dataset/runs/siamese_s120 --vlm_change_cluster_threshold 0.5'
#     CONFIG_NAME  step-3 configuration to filter         (default: sam3_0.1_openVocab)
#     OUTPUT_CSV   CSV collecting the scores of every run   (default: ROOT_DIR/filter_scores.csv)
#
# Examples:
#     ./run_filter_sequence.sh /data3/datasets/smart_change/s120 'siamese' '--siamese_model /data2/.../runs/siamese_s120'
#     ./run_filter_sequence.sh /data3/datasets/smart_change/s120 'vlm_change'
#     ./run_filter_sequence.sh /data3/datasets/smart_change/s120 'siamese vlm_change' '--siamese_model /data2/.../runs/siamese_s120'
#     ./run_filter_sequence.sh /data3/datasets/smart_change/s120 'siamese vlm_change' \
#               '--siamese_model /data2/datasets/change_dataset/runs/siamese_s120 --vlm_change_model qwen3.5:35b+nothink+json'
# Output folder per run: <config>/filters/<stages joined by +>[__<VLM model>]/
#   The VLM model name (as the server reports it) is added whenever a VLM stage is in the sequence,
#   so the same sequence run with different VLMs or VLM settings never overwrites itself.
# Every stage reuses its cache in <config>/filters/_cache/, so answers and scores from earlier runs,
# standalone or in other sequences, are not recomputed.
#
# TESTING: to only print what would run, comment out the two lines marked
# "RUN THE COMMAND" in section 4.
################################################################################


################################################################################
# 1. SETTINGS
################################################################################

ROOT_DIR=${1%/}                                     # folder to search (trailing / removed)
STAGES=$2                                           # e.g. 'siamese vlm_change'
FILTER_ARGS=$3                                      # e.g. '--siamese_model /path/to/model'
CONFIG_NAME=${4:-sam3_0.1_openVocab}                # step-3 configuration folder name
OUTPUT_CSV=${5:-$ROOT_DIR/filter_scores.csv}        # scores of every run, one row per prompt

CONDA_ENV=change                                    # conda environment with cluster_filters
FILTER_DIR=${CHANGE_HOME}/cluster_filters/src/cluster_filters    # folder holding run_filters.py, score_filter.py
SAVE_DIR=change2.0_results                          # step-3 output folder inside each run

if [ -z "$ROOT_DIR" ] || [ ! -d "$ROOT_DIR" ] || [ -z "$STAGES" ]; then
    echo "usage: $0 ROOT_DIR STAGES [FILTER_ARGS] [CONFIG_NAME] [OUTPUT_CSV]"
    echo "  e.g. $0 /data3/datasets/smart_change/s120 'siamese vlm_change' '--siamese_model /path/to/model'"
    exit 1
fi

# the two quoted strings -> lists (quotes inside FILTER_ARGS keep a value with spaces together)
read -r -a STAGE_LIST <<< "$STAGES"
eval "EXTRA_ARGS=($FILTER_ARGS)"


################################################################################
# 2. ENVIRONMENT, VLM SERVER CHECK AND OUTPUT NAME
################################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# output folder name: stages joined by '+'
FILTER_NAME=$(IFS=+; echo "${STAGE_LIST[*]}")

# if a VLM stage is in the sequence: the server must be answering, and its model name goes into the folder name
# if [[ " ${STAGE_LIST[*]} " == *" vlm_"* ]]; then
#     MODEL=$(cd "$FILTER_DIR" && python -c "from llm_query_tools import get_model_name; print(get_model_name())" 2>/dev/null)
if [[ " ${STAGE_LIST[*]} " == *" vlm_"* ]]; then
    # model named in FILTER_ARGS (--vlm_change_model NAME)? use it; otherwise ask the server
    MODEL=""
    for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
        [ "${EXTRA_ARGS[$i]}" = "--vlm_change_model" ] && MODEL=${EXTRA_ARGS[$((i + 1))]}
    done
    [ -z "$MODEL" ] && MODEL=$(cd "$FILTER_DIR" && python -c "from llm_query_tools import get_model_name; print(get_model_name())" 2>/dev/null)
    if [ -z "$MODEL" ]; then
        echo "ERROR: the sequence has a VLM stage but the VLM server is not answering - start it first"
        exit 1
    fi
    echo "VLM server model: $MODEL"
    FILTER_NAME="${FILTER_NAME}__${MODEL//[:\/]/_}"
fi
echo "stages: ${STAGE_LIST[*]}   options: ${EXTRA_ARGS[*]}   output folder: filters/$FILTER_NAME"


################################################################################
# 3. HELPER FUNCTIONS
################################################################################

# has_labels CONFIG
#   True if the configuration folder holds at least one <prompt>.labels.json
has_labels() {
    ls "$1"/*.labels.json > /dev/null 2>&1
}


################################################################################
# 4. MAIN LOOP: every finished step-3 configuration below ROOT_DIR
################################################################################

COUNT=0
SCORED=0
FAILED=()

# every run_settings.json of this configuration (written last, so the step-3 run finished)
for SETTINGS in $(find -L "$ROOT_DIR" -path "*/$SAVE_DIR/$CONFIG_NAME/run_settings.json" | sort); do
    CONFIG=$(dirname "$SETTINGS")
    COUNT=$((COUNT + 1))
    echo "=================================================================="
    echo "CONFIG: $CONFIG"

    # 1. filter
    FILTER_CMD=(python "$FILTER_DIR/run_filters.py" "$CONFIG" --stages "${STAGE_LIST[@]}" "${EXTRA_ARGS[@]}" --name "$FILTER_NAME")
    echo "  ${FILTER_CMD[*]}"
    "${FILTER_CMD[@]}" || { FAILED+=("$CONFIG (filter)"); continue; }        # <<< RUN THE COMMAND - comment out to only print

    # 2. score, only if this run has labels
    if ! has_labels "$CONFIG"; then
        echo "  no labels - not scored"
        continue
    fi
    SCORE_CMD=(python "$FILTER_DIR/score_filter.py" "$CONFIG" "$FILTER_NAME" --output_csv "$OUTPUT_CSV")
    echo "  ${SCORE_CMD[*]}"
    "${SCORE_CMD[@]}" && SCORED=$((SCORED + 1)) || FAILED+=("$CONFIG (score)")   # <<< RUN THE COMMAND - comment out to only print
done


################################################################################
# 5. SUMMARY
################################################################################

echo "=================================================================="
echo "$COUNT finished $CONFIG_NAME configurations under $ROOT_DIR, $SCORED scored into $OUTPUT_CSV"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "${#FAILED[@]} failed:"
    printf '  %s\n' "${FAILED[@]}"
fi