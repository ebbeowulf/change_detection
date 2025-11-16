#!/bin/bash

INITIAL_DIR=$1
NEW_DIR_ROOT=$2

COLOR_DIR=${NEW_DIR_ROOT}/color

echo "Prepare 2 - copy original data to new colmap directory"
COLMAP_DIR=${NEW_DIR_ROOT}/colmap_combined
IMAGE_DIR=${NEW_DIR_ROOT}/images_combined
if [[ ! -d $IMAGE_DIR ]]; then
    mkdir $IMAGE_DIR
fi
A1=$(ls $INITIAL_DIR/images/*.png | wc -l)
B1=$(ls $IMAGE_DIR/*.png | wc -l)
if [[ $A1 -ne $B1 ]]; then
    echo "Images from original directory need to be copied over"
    rm $IMAGE_DIR/*
    echo "cp -r $INITIAL_DIR/images/*.png $IMAGE_DIR"
    cp -r $INITIAL_DIR/images/*.png $IMAGE_DIR
fi
rm -rf ${COLMAP_DIR}
mkdir $COLMAP_DIR
echo "cp $INITIAL_DIR/colmap/database.db $COLMAP_DIR"
cp $INITIAL_DIR/colmap/database.db $COLMAP_DIR

echo "Prepare 3 - Copy new data into colmap directory"
A1=$(ls $COLOR_DIR/*.png | wc -l)
B1=$(ls $IMAGE_DIR/new_*.png | wc -l)
if [[ $A1 -ne $B1 ]]; then
    echo "Rotated images not copied to $IMAGE_DIR - copying"

    cd $COLOR_DIR
    for szFile in *.png
    do
        cp ${szFile} $IMAGE_DIR/new_$szFile
    done
fi

echo "Prepare 4 - Create list of new images for processing"
cd $IMAGE_DIR
NEW_IMAGES="$NEW_DIR_ROOT/new_images.txt"
$(ls new_* > $NEW_DIR_ROOT/new_images.txt)
