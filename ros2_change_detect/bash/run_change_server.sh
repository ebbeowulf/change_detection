#!/bin/bash

NERFACTO_DIR=$1
CHANGES_DIR=$2
DEPTH_DIR=depth_rotated

# Remaining arguments (starting from the 4th)
PROMPTS=()
for arg in "${@:3}"; do
  PROMPTS+=("\"$arg\"")
done

# Activate Conda and ROS 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ros2_env

# Source ROS 2 and your workspace
source ~/ros2_ws/install/setup.bash

# Run your node
cmd="~/miniconda3/envs/ros2_env/bin/python ~/ros2_ws/install/ros2_change_detect/lib/ros2_change_detect/change_server ${NERFACTO_DIR} ${CHANGES_DIR} --depth_dir ${DEPTH_DIR} --queries ${PROMPTS[@]}"
echo $cmd
eval $cmd