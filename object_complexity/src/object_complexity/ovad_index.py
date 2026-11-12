#!/usr/bin/env python3
"""Stage 1 (OVAD): index the 2,000 OVAD images and label attribute + category concepts ("wooden chair").

OVAD attributes each object with 1 (yes), 0 (no), or -1 (unknown). OVAD and COCO do not annotate exactly
the same objects, so each image's objects of a category are the union of:
  matched      OVAD and COCO boxes of the same category with IoU >= --iou (attributes from OVAD)
  OVAD-only    OVAD objects with no COCO match (attributes known)
  COCO-only    non-crowd COCO objects with no OVAD match (attributes unknown)

Image label for concept (attribute a, category c):
  positive  some object of c has a = 1
  negative  no object of c, or every object of c has a = 0 (COCO-only objects smaller than
            --min_area pixels are ignored)
  unknown   anything else; left out of labels.csv, so it is never used as positive or negative

Images are split once at random into "train" (draw pool) and "test" (--test_frac). Only concepts with at
least --min_train_pos and --min_test_pos positives and --min_train_neg negatives are kept.

Writes to results_dir: index.csv, labels.csv (positive and negative rows), dataset.json (exhaustive: false),
and ovad_queries.txt (one query per concept, identical to the concept name).
Reads from ovad_root and coco_root only.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def iou(a, b):
    """Overlap of two [x, y, w, h] boxes: intersection area / union area."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def match(ovad_objs, coco_objs, thr):
    """Greedy one-to-one matching by highest IoU. Returns matched (i, j) pairs and the unmatched of each side."""
    pairs = sorted(((iou(o["bbox"], c["bbox"]), i, j) for i, o in enumerate(ovad_objs)
                    for j, c in enumerate(coco_objs)), reverse=True)
    used_o, used_c, matched = set(), set(), []
    for score, i, j in pairs:
        if score < thr:
            break
        if i not in used_o and j not in used_c:
            matched.append((i, j))
            used_o.add(i)
            used_c.add(j)
    return matched, [i for i in range(len(ovad_objs)) if i not in used_o], \
        [j for j in range(len(coco_objs)) if j not in used_c]


def query_text(att, category):
    """'material:wooden/wood' + 'chair' -> 'wooden chair'; 'has' attributes -> 'chair with ...'."""
    value = att["name"].split(":")[-1].split("/")[0].replace("_", " ")
    return f"{category} with {value}" if att["is_has_att"] == "has" else f"{value} {category}"


def main():
    p = argparse.ArgumentParser(description="Index OVAD attribute + category concepts.")
    p.add_argument("--ovad_root", type=str, required=True, help="Folder containing the OVAD annotation file")
    p.add_argument("--coco_root", type=str, required=True, help="COCO root containing val2017 and annotations")
    p.add_argument("--results_dir", type=str, required=True, help="Output folder (e.g. results_ovad)")
    p.add_argument("--ovad_ann", type=str, default="ovad2000.json", help="OVAD annotation file in ovad_root")
    p.add_argument("--coco_ann", type=str, default="annotations/instances_val2017.json", help="COCO val annotation file under coco_root")
    p.add_argument("--val_dir", type=str, default="val2017", help="COCO val image folder under coco_root")
    p.add_argument("--iou", type=float, default=0.5, help="Minimum box overlap to match an OVAD object to a COCO object")
    p.add_argument("--min_area", type=float, default=1024, help="COCO-only objects smaller than this (pixels) do not block a negative")
    p.add_argument("--test_frac", type=float, default=0.5, help="Fraction of images placed in the test split")
    p.add_argument("--split_seed", type=int, default=0, help="Seed for the train/test split")
    p.add_argument("--min_train_pos", type=int, default=60, help="Minimum positives in the train split to keep a concept")
    p.add_argument("--min_test_pos", type=int, default=20, help="Minimum positives in the test split to keep a concept")
    p.add_argument("--min_train_neg", type=int, default=600, help="Minimum negatives in the train split to keep a concept")
    args = p.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    with open(args.ovad_root + "/" + args.ovad_ann) as f:
        ovad = json.load(f)
    with open(args.coco_root + "/" + args.coco_ann) as f:
        coco = json.load(f)

    # ---- Images: take paths and sizes from COCO; split once into train / test ----
    coco_images = {im["id"]: im for im in coco["images"]}
    image_ids = sorted(im["id"] for im in ovad["images"])
    rng = np.random.default_rng(args.split_seed)
    is_test = rng.random(len(image_ids)) < args.test_frac
    row_of = {img: r for r, img in enumerate(image_ids)}
    split_of = {img: ("test" if t else "train") for img, t in zip(image_ids, is_test)}

    # ---- Objects per (image, category), from both sources ----
    cat_name = {c["id"]: c["name"] for c in ovad["categories"]}
    ovad_by, coco_by = defaultdict(list), defaultdict(list)
    for a in ovad["annotations"]:
        ovad_by[(a["image_id"], a["category_id"])].append(a)
    for a in coco["annotations"]:
        if a["image_id"] in row_of and a["iscrowd"] == 0:  # crowd regions carry no attributes; ignored
            coco_by[(a["image_id"], a["category_id"])].append(a)

    # objects[(image, category)] = list of (attribute vector or None if unknown, area in pixels)
    objects = defaultdict(list)
    for key in set(ovad_by) | set(coco_by):
        o_list, c_list = ovad_by.get(key, []), coco_by.get(key, [])
        matched, ovad_only, coco_only = match(o_list, c_list, args.iou)
        for i, j in matched:
            objects[key].append((np.array(o_list[i]["att_vec"]), c_list[j]["area"]))
        for i in ovad_only:
            objects[key].append((np.array(o_list[i]["att_vec"]), o_list[i]["area"]))
        for j in coco_only:
            objects[key].append((None, c_list[j]["area"]))

    images_with = defaultdict(list)  # category -> images containing at least one object of it
    for img, cat in objects:
        images_with[cat].append(img)

    # ---- Label every (attribute, category) concept; keep those with enough data ----
    concepts, label_rows, seen = [], [], set()
    for cat, imgs in sorted(images_with.items()):
        imgs_set = set(imgs)
        for att in ovad["attributes"]:
            a = att["id"]
            labels = {}  # image -> ("positive", fraction) or ("negative", "")
            for img in imgs:
                objs = objects[(img, cat)]
                im = coco_images[img]
                pos_area = sum(area for vec, area in objs if vec is not None and vec[a] == 1)
                if pos_area > 0:
                    labels[img] = ("positive", round(min(pos_area / (im["width"] * im["height"]), 1.0), 6))
                    continue
                unknown = any((vec is None and area >= args.min_area) or (vec is not None and vec[a] == -1)
                              for vec, area in objs)
                if not unknown:
                    labels[img] = ("negative", "")
            for img in image_ids:  # images without any object of this category are negatives
                if img not in imgs_set:
                    labels[img] = ("negative", "")

            count = defaultdict(int)
            for img, (status, _) in labels.items():
                count[(split_of[img], status)] += 1
            if (count[("train", "positive")] < args.min_train_pos or count[("test", "positive")] < args.min_test_pos
                    or count[("train", "negative")] < args.min_train_neg):
                continue
            name = query_text(att, cat_name[cat])
            if name in seen:  # two attributes with the same wording; keep the first
                print(f"skipping duplicate concept name: {name} ({att['name']})")
                continue
            seen.add(name)
            concepts.append(name)
            for img, (status, frac) in labels.items():
                label_rows.append([row_of[img], name, frac, status])

    # ---- Write outputs ----
    with open(args.results_dir + "/index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "path", "split", "width", "height", "group"])
        for img in image_ids:
            im = coco_images[img]
            path = os.path.abspath(args.coco_root + "/" + args.val_dir + "/" + im["file_name"])
            w.writerow([row_of[img], path, split_of[img], im["width"], im["height"], img])
    with open(args.results_dir + "/labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "category", "pixel_fraction", "status"])
        w.writerows(sorted(label_rows))
    n_test = int(is_test.sum())
    meta = {
        "dataset": "ovad",
        "exhaustive": False,  # images missing from labels.csv for a concept are unknown, not negative
        "categories": concepts,
        "counts": {"train": {"indexed": len(image_ids) - n_test}, "test": {"indexed": n_test}},
        "group": "COCO image id",
    }
    with open(args.results_dir + "/dataset.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(args.results_dir + "/ovad_queries.txt", "w") as f:
        f.write("\n".join(concepts) + "\n")
    print(f"{len(image_ids)} images ({n_test} test), {len(concepts)} concepts kept, {len(label_rows)} labels")


if __name__ == "__main__":
    main()