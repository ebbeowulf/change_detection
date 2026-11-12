#!/usr/bin/env python3
"""Stage 1 (COCO 2017): write index.csv, labels.csv, and dataset.json to the results folder.

index.csv   : row, path, split, width, height, group   (one row per image found on disk)
labels.csv  : row, category, pixel_fraction, status    (one row per image/category present)
dataset.json: dataset name, exhaustive flag, category vocabulary, counts

COCO labels are exhaustive for its 80 categories, so only positives are written;
a category missing from labels.csv for an image means negative.
Reads from coco_root only; writes only to results_dir.
"""
import argparse
import csv
import json
import os
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description="Index COCO 2017 images and frame-level category labels.")
    p.add_argument("--coco_root", type=str, required=True, help="COCO root containing the image folders and annotations folder")
    p.add_argument("--results_dir", type=str, required=True, help="Output folder for index.csv, labels.csv, dataset.json")
    p.add_argument("--ann_dir", type=str, default="annotations", help="Annotations folder name under coco_root")
    p.add_argument("--train_dir", type=str, default="train2017", help="Train image folder name under coco_root")
    p.add_argument("--val_dir", type=str, default="val2017", help="Val image folder name under coco_root")
    p.add_argument("--train_ann", type=str, default="instances_train2017.json", help="Train instance annotation file in ann_dir")
    p.add_argument("--val_ann", type=str, default="instances_val2017.json", help="Val instance annotation file in ann_dir")
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    ann_path = args.coco_root + "/" + args.ann_dir
    splits = [
        (args.train_dir, args.coco_root + "/" + args.train_dir, ann_path + "/" + args.train_ann),
        (args.val_dir, args.coco_root + "/" + args.val_dir, ann_path + "/" + args.val_ann),
    ]

    index_rows, label_rows, counts = [], [], {}
    categories = None
    row = 0
    for split, img_dir, ann_file in splits:
        print(f"Reading {ann_file}")
        with open(ann_file) as f:
            coco = json.load(f)
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        if categories is None:
            categories = [cat_names[k] for k in sorted(cat_names)]

        # Total annotated area per (image, category); includes crowd regions.
        area = defaultdict(float)
        for a in coco["annotations"]:
            area[(a["image_id"], a["category_id"])] += a["area"]
        per_image = defaultdict(list)
        for (img_id, cat_id), total in area.items():
            per_image[img_id].append((cat_id, total))

        missing = 0
        for im in sorted(coco["images"], key=lambda x: x["id"]):
            path = os.path.abspath(img_dir + "/" + im["file_name"])
            if not os.path.isfile(path):
                missing += 1
                continue
            w, h = im["width"], im["height"]
            index_rows.append([row, path, split, w, h, im["id"]])
            for cat_id, total in sorted(per_image[im["id"]]):
                frac = min(total / (w * h), 1.0)  # overlapping instances can overcount
                label_rows.append([row, cat_names[cat_id], round(frac, 6), "positive"])
            row += 1
        counts[split] = {"indexed": len(coco["images"]) - missing, "missing_on_disk": missing}
        print(f"{split}: {counts[split]['indexed']} images indexed, {missing} missing on disk")

    with open(args.results_dir + "/index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "path", "split", "width", "height", "group"])
        w.writerows(index_rows)
    with open(args.results_dir + "/labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "category", "pixel_fraction", "status"])
        w.writerows(label_rows)
    meta = {
        "dataset": "coco2017",
        "exhaustive": True,
        "categories": categories,
        "pixel_fraction": "sum of instance areas (incl. crowd) / image area, clipped to 1",
        "group": "COCO image id",
        "counts": counts,
    }
    with open(args.results_dir + "/dataset.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(index_rows)} images and {len(label_rows)} labels to {args.results_dir}")


if __name__ == "__main__":
    main()