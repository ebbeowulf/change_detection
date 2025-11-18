----------------- How to install ---------------------
Create a conda environment, setup nerf:

conda create -n nerfstudio python=3.8
conda activate nerfstudio
pip install nerfstudio==1.1.0 # this is the version I used, but you can try newer versions as well

Install the python package:
pip install -e .

or to update:

pip install -e . --force-reinstall --no-deps

----------------- How to use ---------------------
This work requires two different data streams:
1) Initial - the initial recording that we will be creating a NeRF model from
2) Changed - the second recording that we are investigating to find/identify Changed

*** Step 1 - Create the file structure ***:
In the original work, images were generated from a ROS bagfile using the associated rgbd_image_saver.py. 
Resulting files were then stored in the following directory structure

INITIAL_DATA_DIR -
   color - rgb_*.png
   depth - depth_*.png
   images.txt - the pose reported by the ros tf library
CHANGE_DATA_DIR - 
   color - rgb_*.png
   depth - depth_*.png
   images.txt - the pose reported by the ros tf library

Note that it is important that your color images are named rgb_{image #}.png or else 
you will have trouble running some of the following scripts

*** Step 2 - Use the bash utilities to prepare and train the NeRF model ***
cd change_nerf_utils/bash
./create_initial_dir-rgbd.sh ${INITIAL_DATA_DIR} # for robot data with depth

or
./create_initial_dir-specAI.sh ${INITIAL_DATA_DIR} # for data captured using the Spectacular AI app on a phone

Note that these will require that your environment variables are set as follows
export CHANGE_HOME={path to root of github repo}

*** Step 3 - Train the NeRF model ***:
Run one of the following commands from your INITIAL_DATA_DIR to start training
1) ns-train splatfacto --data nerf_colmap
2) ns-train depth-nerfacto --data nerf_colmap

*** Step 4 - Build a set of images from the changed data for change detection ***:
Run the associated bash scripts to prepare the directory and register images.
If you have robot data, then run the following from the bash directory

./register_new_images.sh INITIAL_DATA_DIR/nerf_colmap CHANGE_DATA_DIR

or if you do not have robot data, run

./register_new_images-nodepth.sh INITIAL_DATA_DIR/nerf_colmap CHANGE_DATA_DIR

*** Step 5 - Generate the images from the NeRF model for change detection ***
Run the following command from bash:

./generate_nerfstudio_image.sh NERF_MODEL_DIR CHANGE_DATA_DIR RENDERS_DIR

where 
1) NERF_MODEL_DIR is the path to the trained NeRF model from Step 4, 
2) CHANGE_DATA_DIR is the path to the changed data directory from Step 5, and 
3) RENDERS_DIR is the output directory where you want the rendered images to be stored.





