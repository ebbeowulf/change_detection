#!/bin/bash

QUERY=$1
NUM_PTS=$2
CRITERION=${3:-max} #default to max

ros2 service call /get_top1_cluster stretch_srvs/srv/GetCluster "{main_query: '$QUERY', criterion: '$CRITERION', num_points: ${NUM_PTS}}"