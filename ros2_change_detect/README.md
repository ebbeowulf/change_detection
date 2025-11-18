---------------- How to Install --------------
Setup anaconda environment:
conda create -n ros2_env python=3.10
conda activate ros2_env

Make sure your torch version is installed already for the right version of CUDA (e.g):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install companion libraries:
cd ${CHANGE_HOME}/segmentation_utils
pip install -e .
cd ${CHANGE_HOME}/change_pcloud_utils
pip install -e .

Need to setup inside a ros2 workspace. I used symbolic linking:
ln -s ${CHANGE_HOME}/ros2_change_detect ${ROS2_WS}/src/ros2_change_detect

Build your ros2_ws:
This node needs the strech_srvs package inside ltu_domestic_robots. But otherwise standard build

cd ${ROS2_WS}
colcon build

Run (use bash scripts):
cd ${CHANGE_HOME}/ros2_change_detect/bash
./run_change_server.sh {NERF_OUTPUT_DIR} {CHANGE_DIR} {LIST OF QUERIES}

Example...
./run_change_server.sh /data3/datasets/smart_change/j234/baseline/baseline3/outputs/nerf_colmap/depth-nerfacto/2025-11-15_001811/ /data3/datasets/smart_change/j234/changes/changes5 'general clutter' 'trash items'