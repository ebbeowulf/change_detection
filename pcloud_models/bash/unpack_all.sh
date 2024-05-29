#!/bin/bash

DARRAY=($(ls -d -- /data2/datasets/scannet/scans/*))

cd ../scripts/SensReader/c++

for value in "${DARRAY[@]}"
do
    echo $value
    cmd="./sens $value/*.sens $value/raw_output"
    echo $cmd
    eval $cmd
done