---------------- How to Install --------------
Setup anaconda environment:
conda create -n ros2_env python=3.10

Make sure your torch version is installed already for the right version of CUDA (e.g):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Install companion libraries:
cd ${CHANGE_HOME}/segmentation_utils
pip install -e .
cd ${CHANGE_HOME}/change_pcloud_utils
pip install -e .
