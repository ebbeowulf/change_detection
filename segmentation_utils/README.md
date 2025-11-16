Install the package from this directory:

pip install -e .

Can reinstall easily as follows:

pip install -e . --force-reinstall --no-deps

Theoretically can run tests from the command prompt - but it fails to actually show an image over an ssh connection:

segmentation-utils-cli segment --model yolo --image /path/to/image.jpg --tgt-class "clutter" --threshold 0.2
segmentation-utils-cli list-models
segmentation-utils-cli --help
