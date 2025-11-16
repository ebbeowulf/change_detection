#!/bin/bash

QUERY=$1
NUM_PTS=$2

ros2 service call /get_top1_cluster stretch_srvs/srv/GetCluster "{main_query: '$QUERY', criterion: 'max', num_points: ${NUM_PTS}}"