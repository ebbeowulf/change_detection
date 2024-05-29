#!/bin/bash

# Need to run roscore and register.launch first...

DARRAY=($(ls -d -- /data2/datasets/scannet/scans/*))

cd ../scripts/pcloud_models

for value in "${DARRAY[@]}"
do
    cmd="python publish_and_register.py _root_dir:=$value"
    echo $cmd
    eval $cmd
done