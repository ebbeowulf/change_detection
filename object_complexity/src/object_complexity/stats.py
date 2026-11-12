#!/usr/bin/env python3
"""Stage 3: score profile of every query against every reference category.

For each (query, category) pair, describes the query's cosine scores on images labeled as
containing the category (pos) and on all other images (neg): mean, std, d', and AUC.
No score threshold is used; positives are defined by the reference labels only.

Writes ref_dir/<model>/profile_<query_set>[_tag].csv, one row per (query, category).
Leave-one-out and summary statistics are applied later, at analysis time.
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import models


def label_matrix(ref_dir, categories, n_images, min_fraction):
    """(N, C) int8: 1 = contains, 0 = absent, -1 = present below min_fraction (excluded)."""
    col = {c: j for j, c in enumerate(categories)}
    Y = np.zeros((n_images, len(categories)), dtype=np.int8)
    with open(ref_dir + "/labels.csv") as f:
        for r in csv.DictReader(f):
            frac = float(r["pixel_fraction"]) if r["pixel_fraction"] else 1.0
            Y[int(r["row"]), col[r["category"]]] = 1 if frac >= min_fraction else -1
    return Y


def main():
    text_models = [m for m, s in models.MODELS.items() if s["family"] != "dino"]
    p = argparse.ArgumentParser(description="Per (query, category) score profiles on a reference dataset.")
    p.add_argument("--ref_dir", type=str, required=True, help="Reference results folder (index.csv, labels.csv, dataset.json, <model>/)")
    p.add_argument("--model", type=str, required=True, choices=text_models, help="Query model; must have a text encoder")
    p.add_argument("--query_file", type=str, default="", help="Optional file with one query per line; default is the reference category names")
    p.add_argument("--template", type=str, default="{}", help="Prompt template applied to each query, e.g. 'a photo of a {}'")
    p.add_argument("--baseline_prompt", type=str, default="", help="Optional neutral prompt; its score is subtracted from every image's scores")
    p.add_argument("--min_fraction", type=float, default=0.0, help="Minimum pixel fraction to count as present; smaller instances are excluded")
    p.add_argument("--out_tag", type=str, default="", help="Optional tag appended to the output file name")
    p.add_argument("--device", type=str, default="cuda", help="Torch device for text encoding")
    args = p.parse_args()

    with open(args.ref_dir + "/dataset.json") as f:
        meta = json.load(f)
    assert meta["exhaustive"], "stats.py requires an exhaustively labeled reference set"
    categories = meta["categories"]
    n_index = sum(c["indexed"] for c in meta["counts"].values())

    model_dir = args.ref_dir + "/" + args.model
    emb = models.load_embeddings(model_dir)
    assert emb.shape[0] == n_index, f"{emb.shape[0]} embeddings for {n_index} images: finish embed.py first"
    Y = label_matrix(args.ref_dir, categories, n_index, args.min_fraction)

    # Category spread: 1 - |mean of unit vectors| over its positives (0 = identical, larger = more varied).
    spread = np.full(len(categories), np.nan)
    for j in range(len(categories)):
        pos = Y[:, j] == 1
        if pos.any():
            spread[j] = 1.0 - np.linalg.norm(emb[pos].mean(axis=0))

    if args.query_file:
        with open(args.query_file) as f:
            queries = [q.strip() for q in f if q.strip()]
        set_name = os.path.splitext(os.path.basename(args.query_file))[0]
    else:
        queries, set_name = categories, "categories"

    model, processor = models.load(args.model, args.device)
    texts = [args.template.format(q) for q in queries]
    if args.baseline_prompt:
        texts.append(args.baseline_prompt)
    text = models.encode_texts(args.model, model, processor, texts, args.device).cpu().numpy()
    scores = emb @ text.T  # (N, Q[+1]) cosine similarities
    if args.baseline_prompt:
        scores = scores[:, :-1] - scores[:, -1:]

    lower = [c.lower() for c in categories]
    rows = []
    for qi, q in enumerate(queries):
        for j, c in enumerate(categories):
            pos, neg = Y[:, j] == 1, Y[:, j] == 0
            if not pos.any() or not neg.any():
                continue
            sp, sn = scores[pos, qi], scores[neg, qi]
            pooled = np.sqrt((sp.var() + sn.var()) / 2)
            keep = pos | neg
            auc = roc_auc_score(pos[keep], scores[keep, qi])
            rows.append([q, c, int(q.lower() == lower[j]), int(pos.sum()),
                         float(sp.mean()), float(sp.std()), float(sn.mean()), float(sn.std()),
                         float((sp.mean() - sn.mean()) / pooled), float(auc), float(spread[j])])
        if (qi + 1) % 10 == 0 or qi + 1 == len(queries):
            print(f"{qi + 1}/{len(queries)} queries")

    out_file = model_dir + "/profile_" + set_name + ("_" + args.out_tag if args.out_tag else "") + ".csv"
    with open(out_file, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["query", "category", "is_target", "n_pos", "pos_mean", "pos_std",
                     "neg_mean", "neg_std", "d_prime", "auc", "cat_spread"])
        for r in rows:
            wr.writerow([round(v, 5) if isinstance(v, float) else v for v in r])
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()