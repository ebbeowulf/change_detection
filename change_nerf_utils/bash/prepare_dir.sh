#!/bin/bash

INITIAL_DIR=$1 #this is the nerf data directory that contains colmap/
NEW_DIR_ROOT=$2 #this is the root directory that should contain color/ and depth/ subdirs

ROTATED_DIR=${NEW_DIR_ROOT}/rotated
ROTATED_DEPTH_DIR=${NEW_DIR_ROOT}/depth_rotated

echo "Prepare 1 - checking need for rotated images"
A1=$(ls $ROTATED_DIR/*.png | wc -l)
B1=$(ls $NEW_DIR_ROOT/color/*.png | wc -l)
if [[ $A1 -ne $B1 ]]; then
    echo "Rotated image count not equal to color image count - rerunning rotate script"
    ./rotate_images.sh $NEW_DIR_ROOT/color/ $ROTATED_DIR
fi

A1=$(ls $ROTATED_DEPTH_DIR/*.png | wc -l)
B1=$(ls $NEW_DIR_ROOT/depth/*.png | wc -l)
if [[ $A1 -ne $B1 ]]; then
    echo "Rotated depth image count not equal to color image count - rerunning rotate script on depth images"
    ./rotate_images.sh $NEW_DIR_ROOT/depth/ $ROTATED_DEPTH_DIR
fi

echo "Prepare 2 - copy original data to new colmap directory"
COLMAP_DIR=${NEW_DIR_ROOT}/colmap_combined
IMAGE_DIR=${NEW_DIR_ROOT}/images_combined
if [[ ! -d $IMAGE_DIR ]]; then
    mkdir $IMAGE_DIR
fi
A1=$(ls $INITIAL_DIR/images/*.png | wc -l)
B1=$(ls $IMAGE_DIR/frame_*.png | wc -l)
if [[ $A1 -ne $B1 ]]; then
    echo "Images from original directory need to be copied over"
    # echo "rm $IMAGE_DIR/*"
    rm $IMAGE_DIR/*
    # echo "cp -r $INITIAL_DIR/images/*.png $IMAGE_DIR"
    cp -r $INITIAL_DIR/images/*.png $IMAGE_DIR
fi
rm -rf ${COLMAP_DIR}
mkdir $COLMAP_DIR
cp $INITIAL_DIR/colmap/database.db $COLMAP_DIR

echo "Prepare 3 - Copy new data into colmap directory"
A1=$(ls $ROTATED_DIR/*.png | wc -l)
B1=$(ls $IMAGE_DIR/new_*.png | wc -l)
if [[ $A1 -ne $B1 ]]; then
    echo "Rotated images not copied to $IMAGE_DIR - copying"

    cd $ROTATED_DIR
    for szFile in *.png
    do
        cp ${szFile} $IMAGE_DIR/new_$szFile
    done
fi

echo "Prepare 4 - Create list of new images for processing"
cd $IMAGE_DIR
NEW_IMAGES="$NEW_DIR_ROOT/new_images.txt"
$(ls new_* > $NEW_DIR_ROOT/new_images.txt)
