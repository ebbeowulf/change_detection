#!/bin/bash

SPARSE_DIR=$1
IMAGE_TYPE=$2 # png or jpg

# Cycle through a set of colmap subdirectories and run the 
#   model converter script
# Then find the directory with the greatest number of lines of images
#   and set it to be the new 0 directory

best_dir=""
max_count=0

if [[ -f $SPARSE_DIR/0_orig ]];then
    echo "0_orig directory present - no need for further linking"
fi

for dir in "$SPARSE_DIR//"[0-9]/; do
    image_file="${dir}images.txt"
    echo $image_file
    if [[ ! -f "$image_file" ]]; then
        cmd="colmap model_converter --input_path $dir/ --output_path $dir/ --output_type TXT"
        echo $cmd
        eval $cmd
    fi

    if [[ -f "$image_file" ]]; then
        count=$(grep -c "$IMAGE_TYPE" "$image_file")
        echo "$dir has $count image lines"

        # Track max
        if [[ "$count" -gt "$max_count" ]]; then
            max_count=$count
            best_dir="$dir"
        fi
    else
        echo "Missing $image_file in $dir"
    fi
done

echo "Best directory is: $best_dir"
zero_dir="${SPARSE_DIR}/0/"
if [[ "$best_dir" == "${zero_dir}" ]]; then
    echo "best_dir is the 0 directory - no symbolic link needed"
else
    echo "Moving the 0 directory and creating a symbolic link"
    mv ${zero_dir} "${SPARSE_DIR}/0_orig"
    cmd="ln -s ${best_dir} ${SPARSE_DIR}/0"
    echo $cmd
    eval $cmd
fi
