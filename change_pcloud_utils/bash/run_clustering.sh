#!/bin/bash
################################################################################
# run_all_candidates.sh
#
# Re-run build_change_candidates.py on every robot change run under one
# environment folder (for example /data3/datasets/smart_change/s120).
#
# Usage:
#     ./run_all_candidates.sh ROOT_DIR [CLASSIFIER] [THRESHOLD] [QUERIES]
#
#     ROOT_DIR     environment folder to search, e.g. /data3/datasets/smart_change/s120
#     CLASSIFIER   sam3 or clipseg                    (default: sam3)
#     THRESHOLD    candidate threshold                (default: 0.1)
#     QUERIES      prompts as ONE single-quoted string, each prompt in double quotes
#                  (default: the program's own list)
#
# Example:
#     ./run_all_candidates.sh /data3/datasets/smart_change/s120 sam3 0.1 '"trash" "general clutter" "small items"'
#
# For every change run it prints the INITIAL (baseline) folder, the nerfstudio
# CONFIG folder inside it, and the RUN folder, then runs build_change_candidates
# twice: once with --candidates openVocab and once with --candidates change.
# A run/mode that already finished (its run_settings.json exists) is skipped
# with a WARNING; unfinished ones are run.
#
# TESTING: to only print the INITIAL/RUN pairs without processing anything,
# comment out the single line marked "RUN THE COMMAND" in section 4.
################################################################################


################################################################################
# 1. SETTINGS
################################################################################

ROOT_DIR=${1%/}                 # environment folder (trailing / removed)
CLASSIFIER=${2:-sam3}           # segmentation model
THRESHOLD=${3:-0.1}             # candidate threshold
# QUERIES: '"trash" "general clutter"' -> array ("trash" "general clutter"); multi-word prompts stay whole
eval "QUERIES=($4)"

CONDA_ENV=change                # conda environment with change_pcloud_utils installed
MODES=(openVocab change)        # --candidates settings to run for every change run
SAVE_DIR=change2.0_results      # must match build_change_candidates' --save_dir default
# note: THRESHOLD must be written the way Python prints it (0.1, not 0.10) for the
#       finished-run check below to find the output folder

if [ -z "$ROOT_DIR" ] || [ ! -d "$ROOT_DIR" ]; then
    echo "usage: $0 ROOT_DIR [CLASSIFIER] [THRESHOLD] [QUERIES]"
    exit 1
fi


################################################################################
# 2. ENVIRONMENT
################################################################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV


################################################################################
# 3. HELPER FUNCTIONS
################################################################################

# is_change_run DIR
#   True if DIR holds everything build_change_candidates needs:
#   images_combined/, renders/ and colmap_combined/sparse_combined/0/
is_change_run() {
    [ -d "$1/images_combined" ] && [ -d "$1/renders" ] && [ -d "$1/colmap_combined/sparse_combined/0" ]
}

# find_config RUN_DIR
#   Prints the nerfstudio config folder for a change run.
#   Starting at the run's parent, it walks up the folder tree until it finds a
#   folder containing initial/, then searches inside initial/ for
#   .../outputs/.../dataparser_transforms.json.
#   Example: RUN_DIR = s120/T1/changes/changes1
#            -> s120/T1/initial/baseline3/outputs/nerf_colmap/depth-nerfacto/2025-12-09_051107
#   If several configs exist (several training timestamps), the last one by
#   name (the most recent timestamp) is used and a warning is printed.
#   Prints nothing if no config is found below ROOT_DIR.
find_config() {
    local dir
    dir=$(dirname "$1")
    while [ "$dir" != "$(dirname "$ROOT_DIR")" ] && [ "$dir" != "/" ]; do
        if [ -d "$dir/initial" ]; then
            local configs
            configs=$(find -L "$dir/initial" -name dataparser_transforms.json -path "*/outputs/*" | sort)
            if [ -n "$configs" ]; then
                if [ "$(echo "$configs" | wc -l)" -gt 1 ]; then
                    echo "    WARNING: several configs under $dir/initial - using the most recent" >&2
                fi
                dirname "$(echo "$configs" | tail -1)"
                return
            fi
        fi
        dir=$(dirname "$dir")
    done
}

# initial_dir CONFIG_DIR
#   Prints the baseline folder a config belongs to: the folder that holds outputs/.
#   e.g. .../T1/initial/baseline3/outputs/... -> .../T1/initial/baseline3
#        .../T3/initial/outputs/...           -> .../T3/initial
initial_dir() {
    echo "${1%%/outputs/*}"
}


################################################################################
# 4. MAIN LOOP: find every change run, pair it with its initial run, process it
################################################################################

COUNT=0
FAILED=()

# every images_combined/ folder below ROOT_DIR, excluding the initial (baseline) runs
for IMAGES in $(find -L "$ROOT_DIR" -type d -name images_combined -not -path "*/initial/*" | sort); do
    RUN=$(dirname "$IMAGES")

    # skip folders that are not complete change runs
    if ! is_change_run "$RUN"; then
        continue
    fi
    COUNT=$((COUNT + 1))

    # pair the change run with its initial run's nerfstudio config
    echo "=================================================================="
    CONFIG=$(find_config "$RUN")
    if [ -z "$CONFIG" ]; then
        echo "RUN:     $RUN"
        echo "INITIAL: (none found - skipped)"
        FAILED+=("$RUN (no initial config)")
        continue
    fi
    echo "INITIAL: $(initial_dir "$CONFIG")"
    echo "CONFIG:  $CONFIG"
    echo "RUN:     $RUN"

    # process the run once per candidate mode
    for MODE in "${MODES[@]}"; do
        # skip this mode if it already finished: run_settings.json is the last file the program writes
        OUT_DIR="$RUN/$SAVE_DIR/${CLASSIFIER}_${THRESHOLD}_${MODE}"
        if [ -f "$OUT_DIR/run_settings.json" ]; then
            echo "  WARNING: $MODE already finished ($OUT_DIR) - skipped"
            continue
        fi

        # the command is built as an array so multi-word queries ("general clutter") stay one argument
        CMD=(python -m change_pcloud_utils.build_change_candidates "$CONFIG" "$RUN"
             --classifier "$CLASSIFIER" --threshold "$THRESHOLD" --candidates "$MODE")
        if [ ${#QUERIES[@]} -gt 0 ]; then
            CMD+=(--queries "${QUERIES[@]}")
        fi
        echo "  $(printf '%q ' "${CMD[@]}")"

        "${CMD[@]}" || FAILED+=("$RUN ($MODE)")        # <<< RUN THE COMMAND - comment out this line to only print
    done
done


################################################################################
# 5. SUMMARY
################################################################################

echo "=================================================================="
echo "$COUNT change runs found under $ROOT_DIR"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "${#FAILED[@]} failed or skipped:"
    printf '  %s\n' "${FAILED[@]}"
fi