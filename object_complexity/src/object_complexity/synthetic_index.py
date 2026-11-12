#!/usr/bin/env python3
"""Index synthetic (inpainted) images: write index.csv, labels.csv, and dataset.json to a results folder.

Expected layout: <synthetic_root>/<category>/<images_dir>/<img_id>_syn.jpg, where <img_id> is the COCO
image id of the background. Category folders are matched to COCO category names exactly, or with
underscores read as spaces ("traffic_light" -> "traffic light"). Categories with fewer than --min_images
images are left out entirely.

index.csv   : row, path, split ("synthetic"), width, height, group (background COCO id), bg_image_id
labels.csv  : row, category, pixel_fraction (blank), status ("positive")
dataset.json: categories kept, image counts per category; exhaustive is false (only positives are listed)
Reads from synthetic_root and coco_results_dir only; writes only to results_dir.
"""
import argparse
import csv
import json
import os

from PIL import Image


def main():
    p = argparse.ArgumentParser(description="Index synthetic images per category.")
    p.add_argument("--synthetic_root", type=str, required=True, help="Folder holding one subfolder per category")
    p.add_argument("--coco_results_dir", type=str, required=True, help="COCO results folder (for the category names in dataset.json)")
    p.add_argument("--results_dir", type=str, required=True, help="Output folder for the synthetic index (e.g. results_synth)")
    p.add_argument("--images_dir", type=str, default="images", help="Image subfolder inside each category folder")
    p.add_argument("--suffix", type=str, default="_syn", help="File name ending before the extension; the part before it is the COCO image id")
    p.add_argument("--min_images", type=int, default=10, help="Leave out categories with fewer images than this")
    args = p.parse_args()

    with open(args.coco_results_dir + "/dataset.json") as f:
        coco_categories = json.load(f)["categories"]
    os.makedirs(args.results_dir, exist_ok=True)

    index_rows, label_rows, counts, skipped = [], [], {}, {}
    row = 0
    for folder in sorted(os.listdir(args.synthetic_root)):
        img_dir = args.synthetic_root + "/" + folder + "/" + args.images_dir
        if not os.path.isdir(img_dir):
            continue
        # Match the folder to a COCO category name.
        if folder in coco_categories:
            category = folder
        elif folder.replace("_", " ") in coco_categories:
            category = folder.replace("_", " ")
        else:
            print(f"skipping folder with no matching COCO category: {folder}")
            continue

        files = sorted(f for f in os.listdir(img_dir)
                       if os.path.splitext(f)[0].endswith(args.suffix)
                       and os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
        if len(files) < args.min_images:
            skipped[category] = len(files)
            continue

        counts[category] = len(files)
        for name in files:
            bg_id = os.path.splitext(name)[0][: -len(args.suffix)]  # "<img_id>_syn" -> "<img_id>"
            path = os.path.abspath(img_dir + "/" + name)
            with Image.open(path) as im:
                w, h = im.size
            index_rows.append([row, path, "synthetic", w, h, bg_id, bg_id])
            label_rows.append([row, category, "", "positive"])
            row += 1

    with open(args.results_dir + "/index.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "path", "split", "width", "height", "group", "bg_image_id"])
        wr.writerows(index_rows)
    with open(args.results_dir + "/labels.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "category", "pixel_fraction", "status"])
        wr.writerows(label_rows)
    meta = {"dataset": "coco_synthetic", "exhaustive": False, "categories": sorted(counts),
            "counts": {"synthetic": {"indexed": len(index_rows)}}, "images_per_category": counts,
            "skipped_below_min": skipped, "min_images": args.min_images}
    with open(args.results_dir + "/dataset.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{len(counts)} categories kept ({len(index_rows)} images); {len(skipped)} below {args.min_images} images")


if __name__ == "__main__":
    main()