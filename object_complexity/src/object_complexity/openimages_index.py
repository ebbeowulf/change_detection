#!/usr/bin/env python3
"""Index Open Images V7 (validation + test) for the classes in a list file.

Splits: Open Images "validation" images form the pool (thresholds are set from its negatives; synthetic
backgrounds should come from it too), and "test" images form the test set (true recall). In the downstream
scripts, use --test_split test --extra_test 0.

Labels: an image is positive for a class if a human verified the class, or any class below it in the
hierarchy (--hierarchy_file), as present (Confidence 1). Every other image counts as a negative: the class
may occasionally be present but unlabeled, which is treated as noise.

To limit embedding time, the test set keeps every image positive for a kept class plus --max_test_neg
other test images chosen at random. All validation images found on disk are kept.

Writes to results_dir: index.csv, labels.csv (positives only), dataset.json, class_report.csv.
Reads from oi_root and the class list only.
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from PIL import Image


def read_class_list(path):
    """Class list file: 'name<TAB>type' per line; '#' lines ignored. Returns [(name, type)]."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            out.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else "new"))
    return out


def read_class_names(path):
    """Boxable class-name file: LabelName (code) -> DisplayName; works with or without a header row."""
    df = pd.read_csv(path, header=None, names=["code", "name"])
    df = df[df.code != "LabelName"]
    return dict(zip(df.code, df.name))


def descendants(hierarchy_path):
    """code -> set of codes below it in the hierarchy (not including itself); empty dict if no file."""
    if not hierarchy_path:
        return {}
    with open(hierarchy_path) as f:
        tree = json.load(f)
    below = {}

    def walk(node):
        found = set()
        for child in node.get("Subcategory", []):
            found |= {child["LabelName"]} | walk(child)
        below[node["LabelName"]] = below.get(node["LabelName"], set()) | found
        return found

    walk(tree)
    return below


def positives_by_code(label_files):
    """code -> set of image ids verified positive (Confidence 1) in any of the label files."""
    pos = {}
    for path in label_files:
        df = pd.read_csv(path, usecols=["ImageID", "LabelName", "Confidence"])
        df = df[df.Confidence == 1]
        for code, ids in df.groupby("LabelName").ImageID:
            pos.setdefault(code, set()).update(ids)
    return pos


def main():
    p = argparse.ArgumentParser(description="Index Open Images V7 validation + test for a list of classes.")
    p.add_argument("--oi_root", type=str, required=True, help="Open Images folder holding the image folders and CSVs")
    p.add_argument("--class_list", type=str, required=True, help="Absolute path to the class list file (name<TAB>type per line)")
    p.add_argument("--results_dir", type=str, required=True, help="Output folder (e.g. results_oi)")
    p.add_argument("--val_dir", type=str, default="validation", help="Validation image folder under oi_root (the pool)")
    p.add_argument("--test_dir", type=str, default="test", help="Test image folder under oi_root (the test set)")
    p.add_argument("--class_file", type=str, default="oidv7-class-descriptions-boxable.csv", help="Boxable class names in oi_root")
    p.add_argument("--val_labels", type=str, default="oidv7-val-annotations-human-imagelabels.csv", help="Validation human-verified labels in oi_root")
    p.add_argument("--test_labels", type=str, default="oidv7-test-annotations-human-imagelabels.csv", help="Test human-verified labels in oi_root")
    p.add_argument("--hierarchy_file", type=str, default="bbox_labels_600_hierarchy.json", help="Class hierarchy in oi_root; empty = no expansion")
    p.add_argument("--min_test_pos", type=int, default=100, help="Drop classes with fewer positive test images than this")
    p.add_argument("--max_test_neg", type=int, default=20000, help="Random test images kept besides the positives")
    p.add_argument("--seed", type=int, default=0, help="Seed for choosing the extra test images")
    p.add_argument("--read_sizes", type=int, default=1, help="1 = read each image's width and height (slower); 0 = write 0")
    args = p.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # ---- Classes: match the list to Open Images codes ----
    code_to_name = read_class_names(args.oi_root + "/" + args.class_file)
    name_to_code = {n.lower(): c for c, n in code_to_name.items()}
    wanted = read_class_list(args.class_list)
    hier = descendants(args.oi_root + "/" + args.hierarchy_file if args.hierarchy_file else "")

    # ---- Images on disk, per split ----
    val_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.oi_root + "/" + args.val_dir) if f.endswith(".jpg"))
    test_ids = sorted(os.path.splitext(f)[0] for f in os.listdir(args.oi_root + "/" + args.test_dir) if f.endswith(".jpg"))
    val_set, test_set = set(val_ids), set(test_ids)
    print(f"images on disk: {len(val_ids)} validation, {len(test_ids)} test")

    # ---- Positives per class (own code plus everything below it in the hierarchy) ----
    pos_code = positives_by_code([args.oi_root + "/" + args.val_labels, args.oi_root + "/" + args.test_labels])
    report, kept = [], []
    for name, ctype in wanted:
        code = name_to_code.get(name.lower())
        if code is None:
            report.append({"class": name, "type": ctype, "code": "", "pool_pos": 0, "test_pos": 0,
                           "kept": False, "reason": "name not in boxable class file"})
            continue
        codes = {code} | hier.get(code, set())
        pos = set().union(*[pos_code.get(c, set()) for c in codes])
        n_pool, n_test = len(pos & val_set), len(pos & test_set)
        ok = n_test >= args.min_test_pos
        report.append({"class": code_to_name[code], "type": ctype, "code": code, "pool_pos": n_pool,
                       "test_pos": n_test, "codes_included": len(codes), "kept": ok,
                       "reason": "" if ok else f"fewer than {args.min_test_pos} test positives"})
        if ok:
            kept.append((code_to_name[code], ctype, pos))

    # ---- Test images: all positives of kept classes + a random sample of the rest ----
    test_pos_any = set().union(*[pos & test_set for _, _, pos in kept]) if kept else set()
    others = sorted(test_set - test_pos_any)
    rng = np.random.default_rng(args.seed)
    extra = set(rng.choice(others, size=min(args.max_test_neg, len(others)), replace=False)) if others else set()
    test_keep = sorted(test_pos_any | extra)

    # ---- Write index.csv ----
    index_rows, row_of = [], {}
    for split, folder, ids in (("validation", args.val_dir, val_ids), ("test", args.test_dir, test_keep)):
        for img in ids:
            path = os.path.abspath(args.oi_root + "/" + folder + "/" + img + ".jpg")
            w = h = 0
            if args.read_sizes:
                with Image.open(path) as im:
                    w, h = im.size
            row_of[img] = len(index_rows)
            index_rows.append([len(index_rows), path, split, w, h, img])
    with open(args.results_dir + "/index.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "path", "split", "width", "height", "group"])
        wr.writerows(index_rows)

    # ---- Write labels.csv (positives only; every other image is a negative) ----
    with open(args.results_dir + "/labels.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row", "category", "pixel_fraction", "status"])
        for name, _, pos in kept:
            for img in sorted(pos):
                if img in row_of:
                    wr.writerow([row_of[img], name, "", "positive"])

    # ---- Write dataset.json and the class report ----
    meta = {"dataset": "openimages_v7", "exhaustive": True,
            "note": "images without a positive label for a class are treated as negatives (assumption)",
            "categories": [n for n, _, _ in kept], "class_types": {n: t for n, t, _ in kept},
            "counts": {"validation": {"indexed": len(val_ids)}, "test": {"indexed": len(test_keep)}},
            "hierarchy_expanded": bool(args.hierarchy_file)}
    with open(args.results_dir + "/dataset.json", "w") as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame(report).to_csv(args.results_dir + "/class_report.csv", index=False)

    missing = [r["class"] for r in report if r["reason"] == "name not in boxable class file"]
    print(f"{len(kept)} of {len(wanted)} classes kept; {len(index_rows)} images indexed "
          f"({len(val_ids)} validation, {len(test_keep)} test)")
    if missing:
        print("names not found in the class file: " + ", ".join(missing))
    print(f"see {args.results_dir}/class_report.csv for counts and reasons")


if __name__ == "__main__":
    main()