#!/usr/bin/env python3
"""Stage 4: learning curves for a small trained head on frozen, cached features.

For every concept (dataset category): draw N labeled positives plus a fixed number of
negatives from the training pool, train a binary MLP head on the cached embeddings,
and evaluate on a fixed test set.

Test set: all images in --test_split plus --extra_test images (whole groups) from the
other splits, fixed by --split_seed. The remaining images form the training pool.

Writes results_dir/<feature_model>/curves[_tag].csv, one row per (concept, N, seed).
N* (positives needed to reach a target precision/recall) is derived from this at analysis time.
"""
import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import models


def label_matrix(results_dir, categories, n_images, min_fraction):
    """(N, C) int8: 1 = contains, 0 = absent, -1 = present below min_fraction (excluded)."""
    col = {c: j for j, c in enumerate(categories)}
    Y = np.zeros((n_images, len(categories)), dtype=np.int8)
    with open(results_dir + "/labels.csv") as f:
        for r in csv.DictReader(f):
            frac = float(r["pixel_fraction"]) if r["pixel_fraction"] else 1.0
            Y[int(r["row"]), col[r["category"]]] = 1 if frac >= min_fraction else -1
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


def metrics(y, s, recall_target, precision_target):
    prec, rec, _ = precision_recall_curve(y, s)
    p_at_r = prec[rec >= recall_target].max()
    ok = prec >= precision_target
    r_at_p = rec[ok].max() if ok.any() else 0.0
    fpr, tpr, _ = roc_curve(y, s)
    spec_at_r = 1.0 - fpr[tpr >= recall_target].min()
    return [roc_auc_score(y, s), average_precision_score(y, s), p_at_r, r_at_p, spec_at_r]

def train_head(X, y, args, seed):
    """Train a 1-hidden-layer binary head on (X, y); returns a scoring function for new features."""
    torch.manual_seed(seed)
    mu, sd = X.mean(dim=0), X.std(dim=0).clamp_min(1e-6)  # standardize with training-draw statistics
    head = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
                         nn.Linear(args.hidden, 1)).to(X.device)
    pos_weight = (y == 0).sum() / (y == 1).sum()  # positives and negatives carry equal total weight
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Xs = (X - mu) / sd
    head.train()
    for _ in range(args.steps):  # full-batch: the training draw is at most a few thousand embeddings
        opt.zero_grad()
        loss_fn(head(Xs).squeeze(1), y).backward()
        opt.step()
    head.eval()

    @torch.no_grad()
    def score(X_new):
        return head((X_new - mu) / sd).squeeze(1).cpu().numpy()
    return score


def main():
    p = argparse.ArgumentParser(description="Learning curves for a trained MLP head on cached features.")
    p.add_argument("--results_dir", type=str, required=True, help="Results folder (index.csv, labels.csv, dataset.json, <model>/)")
    p.add_argument("--feature_model", type=str, required=True, choices=list(models.MODELS), help="Cached features to train on")
    p.add_argument("--concepts", type=str, default="", help="Optional comma-separated subset of concepts; default is all categories")
    p.add_argument("--sizes", type=str, default="1,2,5,10,20,50,100,200,500,1000", help="Comma-separated numbers of training positives")
    p.add_argument("--n_neg", type=int, default=1000, help="Fixed number of training negatives per draw")
    p.add_argument("--seeds", type=int, default=5, help="Random draws per size for N > small_n")
    p.add_argument("--seeds_small", type=int, default=10, help="Random draws per size for N <= small_n")
    p.add_argument("--small_n", type=int, default=10, help="Sizes at or below this get seeds_small draws")
    p.add_argument("--hidden", type=int, default=256, help="Hidden units in the head")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout after the hidden layer")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    p.add_argument("--steps", type=int, default=300, help="Full-batch training steps per draw")
    p.add_argument("--test_split", type=str, default="val2017", help="Split always placed in the test set")
    p.add_argument("--extra_test", type=int, default=20000, help="Additional images moved from other splits into the test set")
    p.add_argument("--split_seed", type=int, default=0, help="Seed for choosing the extra test images")
    p.add_argument("--min_fraction", type=float, default=0.0, help="Minimum pixel fraction to count as present; smaller instances are excluded")
    p.add_argument("--recall_target", type=float, default=0.8, help="Recall level for the precision-at-recall metric")
    p.add_argument("--precision_target", type=float, default=0.9, help="Precision level for the recall-at-precision metric")
    p.add_argument("--out_tag", type=str, default="", help="Optional tag appended to the output file name")
    p.add_argument("--device", type=str, default="cuda", help="Torch device for training")
    args = p.parse_args()

    with open(args.results_dir + "/dataset.json") as f:
        meta = json.load(f)
    assert meta["exhaustive"], "curves.py currently requires exhaustive labels"
    categories = meta["categories"]
    with open(args.results_dir + "/index.csv") as f:
        index = list(csv.DictReader(f))

    model_dir = args.results_dir + "/" + args.feature_model
    emb = models.load_embeddings(model_dir)
    assert emb.shape[0] == len(index), f"{emb.shape[0]} embeddings for {len(index)} images: finish embed.py first"
    Y = label_matrix(args.results_dir, categories, len(index), args.min_fraction)
    test = make_test_mask(index, args.test_split, args.extra_test, args.split_seed)
    print(f"test images: {test.sum()}, training pool: {(~test).sum()}")
    emb_gpu = torch.from_numpy(emb).to(args.device)

    sizes = [int(s) for s in args.sizes.split(",")]
    selected = set(args.concepts.split(",")) if args.concepts else None
    rows = []
    for c, concept in enumerate(categories):
        if selected and concept not in selected:
            continue
        t0 = time.time()
        y = Y[:, c]
        test_c = test & (y >= 0)
        y_test = (y[test_c] == 1).astype(int)
        X_test = emb_gpu[torch.from_numpy(np.where(test_c)[0]).to(args.device)]
        pool_pos = np.where(~test & (y == 1))[0]
        pool_neg = np.where(~test & (y == 0))[0]
        n_test_pos = int(y_test.sum())

        for seed in range(max(args.seeds, args.seeds_small)):
            rng = np.random.default_rng([seed, c])
            perm = rng.permutation(pool_pos)  # positives are nested: N=5 contains the N=2 draw
            neg = rng.choice(pool_neg, size=min(args.n_neg, len(pool_neg)), replace=False)
            for n in sizes:
                n_seeds = args.seeds_small if n <= args.small_n else args.seeds
                if n > len(perm) or seed >= n_seeds:
                    continue
                idx = torch.from_numpy(np.concatenate([perm[:n], neg])).to(args.device)
                yy = torch.cat([torch.ones(n), torch.zeros(len(neg))]).to(args.device)
                score = train_head(emb_gpu[idx], yy, args, seed)
                rows.append([concept, args.feature_model, n, len(neg), seed, n_test_pos, len(y_test)] +
                            metrics(y_test, score(X_test), args.recall_target, args.precision_target))
        print(f"{c + 1}/{len(categories)} {concept}: {len(pool_pos)} pool positives, "
              f"{n_test_pos} test positives, {time.time() - t0:.1f}s")

    out_file = model_dir + "/curves" + ("_" + args.out_tag if args.out_tag else "") + ".csv"
    with open(out_file, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["concept", "feature_model", "n_pos", "n_neg", "seed", "n_test_pos", "n_test",
                     "auc", "ap", f"p_at_r{args.recall_target}", f"r_at_p{args.precision_target}",
                     f"spec_at_r{args.recall_target}"])        
        for r in rows:
            wr.writerow([round(float(v), 5) if isinstance(v, (float, np.floating)) else v for v in r])
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()