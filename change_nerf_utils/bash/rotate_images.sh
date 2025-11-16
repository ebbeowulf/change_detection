#!/bin/bash

TGT_DIR=$1
SAVE_DIR=$2

mkdir ${SAVE_DIR}

for szFile in $TGT_DIR/*.png
do
    convert "$szFile" -rotate 90 $SAVE_DIR/"$(basename "$szFile")" ;
done
