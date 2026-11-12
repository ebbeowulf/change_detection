#!/bin/bash
# Run recall_error.py on two GPUs, interleaving concepts between them.
# Usage: ./run_recall_error.sh <results_dir> <feature_model>
RESULTS_DIR=$1
FEATURE_MODEL=$2
LOG_DIR=$RESULTS_DIR/$FEATURE_MODEL

PYTHON_HOME=$CHANGE_HOME/object_complexity/src/object_complexity
cd $PYTHON_HOME

python recall_error.py --results_dir $RESULTS_DIR --feature_model $FEATURE_MODEL \
    --shard 0 --num_shards 2 --device cuda:0 --out_tag shard0 > $LOG_DIR/recall_error_shard0.log 2>&1 &
python recall_error.py --results_dir $RESULTS_DIR --feature_model $FEATURE_MODEL \
    --shard 1 --num_shards 2 --device cuda:1 --out_tag shard1 > $LOG_DIR/recall_error_shard1.log 2>&1 &
wait
echo "Done: $LOG_DIR/recall_error_shard0.csv and recall_error_shard1.csv"