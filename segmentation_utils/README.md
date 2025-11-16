-------------- How to install -------------
Make sure your torch version is installed already
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Install the package from this directory:

pip install -e .

Can reinstall easily as follows:

pip install -e . --force-reinstall --no-deps

-------------- Usage -------------
Theoretically can run tests from the command prompt - but it fails to actually show an image over an ssh connection:

segmentation-utils-cli segment --model yolo --image /path/to/image.jpg --tgt-class "clutter" --threshold 0.2
segmentation-utils-cli list-models
segmentation-utils-cli --help

Otherwise can process individual files to check for segmentation:
python clip_segmentation.py {IMAGE_NAME} {query}
python dino_segmentation.py {IMAGE_NAME} {query}
python yolo_world_segmentation.py {IMAGE_NAME} {query}

Or compare a pair of before and after images:
python test_change_single_pair.py {BEFORE IMAGE} {AFTER IMAGE} {query} 