#!/bin/bash
# Build synthetic positives for COCO categories, one category at a time:
#   rank backgrounds (caption similarity) -> plan (Qwen) -> inpaint + verify (FLUX + Qwen)
# Usage:
#   ./run_synthetic.sh 10 1      # 10 images per category, first category only (test)
#   ./run_synthetic.sh 10 0      # 10 images per category, all categories
# Safe to re-run: existing rankings and plans are kept, and images already in
# inpaint_log.json are not re-inpainted.

NUM_IMAGES=$1
START_CATEGORY_NUM=$2     # 0-based starting category number
END_CATEGORY_NUM=$3       # 0-based ending category number (up to 80)

PYTHON_DIR=${CHANGE_HOME}/llm_evaluation/src/llm_evaluation/image_synthesis
COCO_DIR=/data3/datasets/coco/coco
NEG_DIR=$COCO_DIR/train2017
ANN_DIR=$COCO_DIR/annotations
CAPTION_DIR=/data3/datasets/coco/scene_captions
OUT_ROOT=/data3/datasets/coco/synthetic_ranked
PLAN_FACTOR=3         # plan this many times more images than needed, to allow for failures
EXCLUDE=""            # e.g. "kite|frisbee" to skip categories

cd "$PYTHON_DIR" || exit 1
mkdir -p "$OUT_ROOT"
python -c "import json; [print(c['name']) for c in json.load(open('$ANN_DIR/instances_train2017.json'))['categories']]" > "$OUT_ROOT/categories.txt"
echo "$EXCLUDE" | tr '|' '\n' > "$OUT_ROOT/excluded_categories.txt"

COUNT=0
while read -r CATEGORY; do
    COUNT=$((COUNT + 1))
    if echo "$CATEGORY" | grep -qxE "$EXCLUDE"; then
        continue
    fi
    if [ "$COUNT" -le "$START_CATEGORY_NUM" ]; then
        continue
    fi
    if [ "$COUNT" -ge "$END_CATEGORY_NUM" ]; then
        break
    fi
    CAT_DIR="$OUT_ROOT/${CATEGORY// /_}"

    echo "=== $COUNT: $CATEGORY  inpaint"
    if [ -f "$CAT_DIR/plan.json" ]; then
        python inpaint_boxes.py --plan_dir "$CAT_DIR" --num_images "$NUM_IMAGES" < /dev/null
    else
        echo "    no plan.json (planning failed), skipping inpainting"
    fi
done < "$OUT_ROOT/categories.txt"