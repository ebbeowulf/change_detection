#!/usr/bin/env python3
"""Zero-shot recall from synthetic images, compared with the earlier zero-shot predictors.

For each COCO category with synthetic images, using the open-vocabulary score f(img, Q), Q = its name:
  threshold     the score that a given share (the specificity level) of the real training-pool images
                WITHOUT the category fall below (the same thresholds as basket_recall.py)
  synthetic     share of the category's synthetic images scoring above the threshold
  true recall   share of the category's real test-set images scoring above the threshold

Compared on the same categories and thresholds:
  mean          average true recall of the other 79 COCO categories
  size_ridge, size_forest   target's estimated size only, trained on the other 79 categories
  acc_forest, acc+size_forest   from basket_recall output (--basket_file), averaged over the baskets
                where the category was a target; only categories with a basket prediction are used

Reports, in recall units, per specificity level: typical |predicted - true recall|, average signed error
(predicted - true; positive = too optimistic), and the error that 90% of categories stay within.
Writes synth_dir/<model>/synth_recall.csv and, with --plot 1, plots/synth_recall.png.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")  # write files only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import models


def label_matrix(results_dir, categories, n_images):
    """(images x categories) table: 1 = image contains the category, 0 = it does not (exhaustive labels)."""
    col = {c: j for j, c in enumerate(categories)}
    Y = np.zeros((n_images, len(categories)), dtype=np.int8)
    with open(results_dir + "/labels.csv") as f:
        for r in csv.DictReader(f):
            Y[int(r["row"]), col[r["category"]]] = 1
    return Y


def make_test_mask(index, test_split, extra_test, seed):
    """Test = all of test_split + whole groups from other splits until extra_test images are added."""
    split = np.array([r["split"] for r in index])
    group = np.array([r["group"] for r in index])
    test = split == test_split
    if extra_test <= 0:
        return test
    groups, counts = np.unique(group[~test], return_counts=True)
    order = np.random.default_rng(seed).permutation(len(groups))
    n_needed = np.searchsorted(np.cumsum(counts[order]), extra_test) + 1
    test |= np.isin(group, groups[order[:n_needed]])
    return test


def summarize(pred, true):
    """Typical |error|, average signed error, and the error 90% of predictions stay within."""
    e = pred - true
    return [np.abs(e).mean(), e.mean(), np.quantile(np.abs(e), 0.9)]


def main():
    text_models = [m for m, s in models.MODELS.items() if s["family"] != "dino"]
    p = argparse.ArgumentParser(description="Raw synthetic recall versus earlier zero-shot predictors.")
    p.add_argument("--results_dir", type=str, required=True, help="Real COCO results folder (index.csv, labels.csv, dataset.json, <model>/)")
    p.add_argument("--synth_dir", type=str, required=True, help="Synthetic results folder from synthetic_index.py and embed.py")
    p.add_argument("--model", type=str, required=True, choices=text_models, help="Open-vocabulary model providing the scores")
    p.add_argument("--sizes_file", type=str, default="object_sizes.json", help="Object size lookup table in results_dir")
    p.add_argument("--size_key", type=str, default="image_fraction", help="Which size value to use: image_fraction or surface_area_m2")
    p.add_argument("--basket_file", type=str, default="basket_recall_size.csv", help="basket_recall output in results_dir/<model>; empty = skip")
    p.add_argument("--specs", type=str, default="0.8,0.9,0.95,0.99", help="Comma-separated specificity levels for the threshold")
    p.add_argument("--template", type=str, default="{}", help="Prompt template applied to each category name")
    p.add_argument("--alpha", type=float, default=10.0, help="Ridge penalty on large weights")
    p.add_argument("--trees", type=int, default=200, help="Trees in the random forest")
    p.add_argument("--test_split", type=str, default="val2017", help="Split always placed in the test set")
    p.add_argument("--extra_test", type=int, default=20000, help="Additional images moved from other splits into the test set")
    p.add_argument("--split_seed", type=int, default=0, help="Seed for the test split")
    p.add_argument("--plot", type=int, default=1, help="1 = write a synthetic-vs-true recall plot")
    args = p.parse_args()

    # ---- Real COCO: labels, embeddings, test split ----
    with open(args.results_dir + "/dataset.json") as f:
        categories = json.load(f)["categories"]
    C = len(categories)
    with open(args.results_dir + "/index.csv") as f:
        index = list(csv.DictReader(f))
    emb = models.load_embeddings(args.results_dir + "/" + args.model)
    assert emb.shape[0] == len(index), "COCO embeddings incomplete: finish embed.py first"
    Y = label_matrix(args.results_dir, categories, len(index))
    test = make_test_mask(index, args.test_split, args.extra_test, args.split_seed)
    pool = ~test
    test_ids = {r["group"] for r, t in zip(index, test) if t}  # COCO image ids in the real test set

    # ---- Synthetic images: index and embeddings ----
    with open(args.synth_dir + "/dataset.json") as f:
        synth_meta = json.load(f)
    with open(args.synth_dir + "/index.csv") as f:
        synth_index = list(csv.DictReader(f))
    with open(args.synth_dir + "/labels.csv") as f:
        synth_cat = {int(r["row"]): r["category"] for r in csv.DictReader(f)}
    synth_emb = models.load_embeddings(args.synth_dir + "/" + args.model)
    assert synth_emb.shape[0] == len(synth_index), "synthetic embeddings incomplete: run embed.py on synth_dir"
    synth_rows = {c: [i for i in range(len(synth_index)) if synth_cat[i] == c] for c in synth_meta["categories"]}

    # ---- Scores, thresholds, true and synthetic recall ----
    model, processor = models.load(args.model, "cuda")
    text = models.encode_texts(args.model, model, processor,
                               [args.template.format(c) for c in categories], "cuda").cpu().numpy()
    specs = [float(s) for s in args.specs.split(",")]
    thr = np.zeros((C, len(specs)))
    true_recall = np.zeros((C, len(specs)))
    for c in range(C):
        s = emb @ text[c]
        neg_scores = s[pool & (Y[:, c] == 0)]
        for j, spec in enumerate(specs):
            thr[c, j] = np.quantile(neg_scores, spec)
            true_recall[c, j] = (s[test & (Y[:, c] == 1)] > thr[c, j]).mean()

    # ---- Earlier predictors that need no basket: leave one category out over all 80 ----
    with open(args.results_dir + "/" + args.sizes_file) as f:
        size_table = json.load(f)
    log_size = np.log([size_table[c][args.size_key] for c in categories])
    fits = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=args.alpha)),
            "forest": lambda: RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=3,
                                                    max_features=0.5, n_jobs=-1, random_state=0)}
    cat_idx = {c: i for i, c in enumerate(categories)}
    targets = [cat_idx[c] for c in synth_meta["categories"]]
    preds = {"mean": np.zeros((C, len(specs))), "size_ridge": np.zeros((C, len(specs))),
             "size_forest": np.zeros((C, len(specs))), "synthetic": np.zeros((C, len(specs)))}
    for j in range(len(specs)):
        for c in targets:
            train = np.arange(C) != c
            preds["mean"][c, j] = true_recall[train, j].mean()
            for name, make in fits.items():
                fitted = make().fit(log_size[train][:, None], true_recall[train, j])
                preds["size_" + name][c, j] = np.clip(fitted.predict(log_size[c:c + 1][:, None])[0], 0, 1)

    # ---- Synthetic recall, and how many backgrounds came from the real test set ----
    n_synth, n_bg_test = {}, {}
    for c in targets:
        rows = synth_rows[categories[c]]
        s = synth_emb[rows] @ text[c]
        n_synth[c] = len(rows)
        n_bg_test[c] = sum(synth_index[i]["bg_image_id"] in test_ids for i in rows)
        for j in range(len(specs)):
            preds["synthetic"][c, j] = (s > thr[c, j]).mean()

    # ---- Basket predictions (optional), averaged over the baskets where the category was a target ----
    basket_names = []
    basket_path = args.results_dir + "/" + args.model + "/" + args.basket_file
    if args.basket_file and os.path.exists(basket_path):
        b = pd.read_csv(basket_path)
        for name in ("acc_forest", "acc+size_forest"):
            if "pred_" + name in b.columns:
                basket_names.append(name)
                preds[name] = np.full((C, len(specs)), np.nan)
                avg = b.groupby(["concept", "spec"])["pred_" + name].mean()
                for (concept, spec), v in avg.items():
                    if concept in cat_idx and spec in specs:
                        preds[name][cat_idx[concept], specs.index(spec)] = v
    elif args.basket_file:
        print(f"no basket file at {basket_path}; basket comparison skipped")

    # ---- Report ----
    print(f"\n{len(targets)} categories with synthetic images "
          f"({min(n_synth.values())}-{max(n_synth.values())} images each; "
          f"{sum(n_bg_test.values())} of {sum(n_synth.values())} backgrounds are real test-set images)")
    names = ["mean", "size_ridge", "size_forest", "synthetic"]
    for title, k in (("Typical |predicted - true recall|", 0), ("Average signed error (predicted - true)", 1),
                     ("90% of categories within", 2)):
        print(f"\n{title}:")
        print(f"{'':>16}" + "".join(f"{'spec=' + str(s):>11}" for s in specs))
        for n in names:
            vals = [summarize(preds[n][targets, j], true_recall[targets, j])[k] for j in range(len(specs))]
            print(f"{n:>16}" + "".join(f" {v:+10.3f}" if k == 1 else f" {v:10.3f}" for v in vals))

    if basket_names:  # basket models only cover the categories that were basket targets
        have = [c for c in targets if not np.isnan(preds[basket_names[0]][c]).any()]
        print(f"\nOn the {len(have)} categories that also have basket predictions:")
        for title, k in (("Typical |predicted - true recall|", 0), ("90% of categories within", 2)):
            print(f"\n{title}:")
            print(f"{'':>16}" + "".join(f"{'spec=' + str(s):>11}" for s in specs))
            for n in names + basket_names:
                vals = [summarize(preds[n][have, j], true_recall[have, j])[k] for j in range(len(specs))]
                print(f"{n:>16}" + "".join(f" {v:10.3f}" for v in vals))

    # ---- Save ----
    out_dir = args.synth_dir + "/" + args.model
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for c in targets:
        for j, spec in enumerate(specs):
            rows.append({"concept": categories[c], "spec": spec, "threshold": thr[c, j],
                         "true_recall": true_recall[c, j], "n_synthetic": n_synth[c],
                         "n_bg_in_test": n_bg_test[c], **{"pred_" + n: preds[n][c, j] for n in preds}})
    out = pd.DataFrame(rows)
    out_file = out_dir + "/synth_recall.csv"
    out.round(4).to_csv(out_file, index=False)
    print(f"\nWrote {out_file}")

    # ---- Optional plot: synthetic vs true recall, colored by specificity ----
    if args.plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(out.pred_synthetic, out.true_recall, c=out.spec, cmap="viridis", s=12)
        ax.plot([0, 1], [0, 1], color="red", linewidth=1)  # perfect prediction
        ax.set_xlabel("synthetic recall")
        ax.set_ylabel("true recall")
        fig.colorbar(sc, ax=ax, label="specificity level")
        os.makedirs(out_dir + "/plots", exist_ok=True)
        fig.savefig(out_dir + "/plots/synth_recall.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_dir}/plots/synth_recall.png")


if __name__ == "__main__":
    main()