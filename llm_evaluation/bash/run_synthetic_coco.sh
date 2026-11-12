#!/bin/bash
# Build synthetic positives for COCO categories, one category at a time:
#   rank backgrounds -> plan (Qwen) -> inpaint + verify (FLUX + Qwen)
# If a category ends up with fewer than NUM_IMAGES verified images, it is re-planned
# with a larger plan (the first entries come out the same) and inpainting resumes,
# for up to MAX_ROUNDS rounds.
# Usage:
#   ./run_synthetic.sh 10 1      # 10 images per category, first category only (test)
#   ./run_synthetic.sh 10 0      # 10 images per category, all categories

NUM_IMAGES=$1
START_CATEGORY_NUM=$2     # 0-based starting category number
END_CATEGORY_NUM=$3       # 0-based ending category number (up to 80)

PYTHON_DIR=${CHANGE_HOME}/llm_evaluation/src/llm_evaluation/image_synthesis
COCO_DIR=/data3/datasets/coco/coco
NEG_DIR=$COCO_DIR/train2017
ANN_DIR=$COCO_DIR/annotations
CAPTION_DIR=/data3/datasets/coco/scene_captions
OUT_ROOT=/data3/datasets/coco/synthetic_ranked
PLAN_FACTOR=2         # round r plans NUM_IMAGES * PLAN_FACTOR * r images
MAX_ROUNDS=3          # plan/inpaint rounds per category before giving up
EXCLUDE=""            # e.g. "kite|frisbee" to skip categories

cd "$PYTHON_DIR" || exit 1
mkdir -p "$OUT_ROOT"
python -c "import json; [print(c['name']) for c in json.load(open('$ANN_DIR/instances_train2017.json'))['categories']]" > "$OUT_ROOT/categories.txt"
echo "$EXCLUDE" | tr '|' '\n' > "$OUT_ROOT/excluded_categories.txt"
SUMMARY="$OUT_ROOT/summary.txt"

count_verified() {   # number of verified images in a category folder
    python -c "import json, os; f='$1/inpaint_log.json'; print(sum(v['verified'] for v in json.load(open(f))['images'].values()) if os.path.exists(f) else 0)"
}
count_planned() {    # number of entries in a category's plan.json
    python -c "import json, os; f='$1/plan.json'; print(len(json.load(open(f))['entries']) if os.path.exists(f) else 0)"
}

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

    if [ -f "$CAT_DIR/incomplete.txt" ]; then
        echo "=== $CATEGORY  marked incomplete on an earlier run, skipping"
        continue
    fi

    echo "=== $COUNT: $CATEGORY  rank"
    if [ ! -f "$CAT_DIR/background_ranking.json" ]; then
        python rank_backgrounds.py --caption_dir "$CAPTION_DIR" --ann_dir "$ANN_DIR" --query "$CATEGORY" \
            --out_dir "$CAT_DIR" < /dev/null
    fi
    if [ ! -f "$CAT_DIR/background_ranking.json" ]; then
        echo "    ranking failed, skipping category"
        continue
    fi

    for ROUND in $(seq 1 $MAX_ROUNDS); do
        VERIFIED=$(count_verified "$CAT_DIR")
        if [ "$VERIFIED" -ge "$NUM_IMAGES" ]; then
            break
        fi
        PLAN_SIZE=$((NUM_IMAGES * PLAN_FACTOR * ROUND))
        PLANNED=$(count_planned "$CAT_DIR")
        echo "=== $COUNT: $CATEGORY  round $ROUND: $VERIFIED verified, $PLANNED planned, target plan $PLAN_SIZE"

        NEW_PLAN=0
        if [ "$PLANNED" -lt "$PLAN_SIZE" ]; then
            NEW_PLAN=1
            if [ -f "$CAT_DIR/plan.json" ]; then
                cp "$CAT_DIR/plan.json" "$CAT_DIR/plan_before_round$ROUND.json"   # keep the previous plan
            fi
            python plan_insertions.py --neg_dir "$NEG_DIR" --ann_dir "$ANN_DIR" --query "$CATEGORY" \
                --out_dir "$CAT_DIR" --num_images "$PLAN_SIZE" < /dev/null
            if [ "$(count_planned "$CAT_DIR")" -le "$PLANNED" ]; then
                echo "    no new backgrounds could be planned, stopping"
                break
            fi
        fi
        python inpaint_boxes.py --plan_dir "$CAT_DIR" --num_images "$NUM_IMAGES" < /dev/null
        if [ "$NEW_PLAN" -eq 1 ] && [ "$(count_verified "$CAT_DIR")" -le "$VERIFIED" ]; then
            echo "    new backgrounds gave no new verified images, stopping"
            break
        fi
    done

    FINAL=$(count_verified "$CAT_DIR")
    if [ "$FINAL" -lt "$NUM_IMAGES" ]; then
        echo "verified $FINAL of $NUM_IMAGES with $(count_planned "$CAT_DIR") planned" > "$CAT_DIR/incomplete.txt"
        echo "    giving up on $CATEGORY: $FINAL of $NUM_IMAGES verified"
    fi
    
    echo "$CATEGORY	$(count_verified "$CAT_DIR")	$(count_planned "$CAT_DIR")" >> "$SUMMARY"
done < "$OUT_ROOT/categories.txt"