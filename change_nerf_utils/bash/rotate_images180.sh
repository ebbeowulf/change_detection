#!/bin/bash

TGT_DIR=$1
SAVE_DIR=$2

mkdir ${SAVE_DIR}

for szFile in $TGT_DIR/*.jpg
do
    convert "$szFile" -rotate 180 $SAVE_DIR/"$(basename "$szFile")" ;
done
