----------------- How to install ---------------------
Create a conda environment, setup transformers and ultralytics:

conda create -n clipseg python=3.10
conda activate clipseg
pip install transformers
pip install ultralytics

Install the python package:
pip install -e .

or to update:

pip install -e . --force-reinstall --no-deps

----------------- How to use ---------------------
In the original work, images were generated from a ROS bagfile using the associated rgbd_image_saver.py. 
Then we ran the bash scripts in change_nerf_utils to create the colmap data structures. Now we should have
something like the following data structure on your disk:

INITIAL_DATA_DIR -
   color - rgb_*.png
   depth - depth_*.png
   images.txt # the pose reported by the ros tf library
   nerf_colmap/ # the nerf directory created in the baseline
   nerf_colmap/transforms.json
   nerf_colmap/colmap/sparse_geo/0/images.txt
   outputs/nerf_colmap/{splatfacto,depthacto}/{TIMESTAMP} #this is your NERFSTUDIO_CONFIG_DIR
CHANGE_DATA_DIR - 
   color - rgb_*.png
   depth - depth_*.png
   images.txt - the pose reported by the ros tf library
   transforms.json
   colmap_combined/sparse_geo/0/images.txt


Visualize your pointclouds:
python change_pcloud_utils/visualize_colmap.py ${NERFSTUDIO_CONFIG_DIR} ${CHANGE_DATA_DIR}

Visualize change detection:
Compare any two files using test_single_pair.py. 
python change_pcloud_utils/test_single_pair.py ${START_IMAGE} ${CHANGE_IMAGE} ${CLIPS_QUERY} --threshold ${FIXED_THRESHOLD}

This always searches for positive change (i.e. CHANGE_IMAGE - START_IMAGE). To reverse, change the order of the inputs. You can also input a percentage threshold that finds change relative to the detected max delta.

python change_pcloud_utils/test_single_pair.py ~/change_detection/plants/weed1/color/rgb_0010.png ~/change_detection/plants/weed1/renders/rgb_0010.png "unwanted weed" --pct_threshold 0.9
