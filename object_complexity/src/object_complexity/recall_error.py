#!/usr/bin/env python3
"""Stage 4: error of recall measured on N labeled positives, at a fixed specificity.

For every concept, N, and random draw: draw N positives and --n_neg negatives, split both once
into training and validation, and train a head on the training split.
  val_recall   recall on the validation positives, at the threshold that gives specificity
               --spec on the validation negatives
  test_recall  the same head and the same threshold, on all test-set positives
  error        val_recall - test_recall

Training uses balanced batches: every step sees all training positives plus a fresh random
subset of negatives (--neg_per_pos per positive). Set --neg_per_pos 0 to use all negatives
every step with total-weight balancing instead (for comparison).

Test set: all images in --test_split plus --extra_test images (whole groups) from the
other splits, fixed by --split_seed. Draws come only from the remaining training pool.

--shard / --num_shards split the concepts across processes (e.g., one per GPU); each
process writes its own file. Output: results_dir/<feature_model>/recall_error[_tag].csv
"""
import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn

import models

def label_matrix(results_dir, categories, n_images, min_fraction, exhaustive):
    """Build a table of labels: one row per image, one column per category (concept).

    Values: 1 = contains it, 0 = does not contain it, -1 = excluded (unknown, or present but smaller
    than min_fraction). exhaustive=True (COCO): images not listed in labels.csv are negatives.
    exhaustive=False (OVAD): only rows marked "negative" count as negatives; unlisted images are unknown.
    """
    col = {c: j for j, c in enumerate(categories)}  # concept name -> column number
    if exhaustive:
        Y = np.zeros((n_images, len(categories)), dtype=np.int8)  # start with "absent" everywhere
    else:
        Y = np.full((n_images, len(categories)), -1, dtype=np.int8)  # start with "unknown" everywhere
    with open(results_dir + "/labels.csv") as f:
        for r in csv.DictReader(f):
            if r["status"] == "negative":
                Y[int(r["row"]), col[r["category"]]] = 0
                continue
            frac = float(r["pixel_fraction"]) if r["pixel_fraction"] else 1.0
            Y[int(r["row"]), col[r["category"]]] = 1 if frac >= min_fraction else -1
    return Y

def make_test_mask(index, test_split, extra_test, seed):
    """Return a True/False array over all images: True = image is in the test set.

    The test set is every image in test_split (val2017) plus about extra_test images
    taken from the other splits. Images are moved in whole groups, so for video
    datasets no scene ends up on both sides.
    """
    split = np.array([r["split"] for r in index])
    group = np.array([r["group"] for r in index])
    test = split == test_split  # start with all of val2017
    if extra_test <= 0:
        return test
    groups, counts = np.unique(group[~test], return_counts=True)  # groups outside val2017, with sizes
    order = np.random.default_rng(seed).permutation(len(groups))  # shuffle them (fixed seed)
    n_needed = np.searchsorted(np.cumsum(counts[order]), extra_test) + 1  # how many groups reach extra_test images
    test |= np.isin(group, groups[order[:n_needed]])  # move those groups into the test set
    return test


def train_head(X_pos, X_neg, args, seed):
    """Train a small network (one hidden layer) to separate positives from negatives.

    X_pos, X_neg: embeddings of the training positives and negatives (on the GPU).
    Returns a function that scores new embeddings; higher score = more likely to contain the concept.
    """
    torch.manual_seed(seed)  # makes each head's training reproducible
    dev = X_pos.device
    # Rescale each embedding dimension to mean 0, std 1, using all training images.
    X_all = torch.cat([X_pos, X_neg])
    mu, sd = X_all.mean(dim=0), X_all.std(dim=0).clamp_min(1e-6)
    Xp, Xn = (X_pos - mu) / sd, (X_neg - mu) / sd

    head = nn.Sequential(nn.Linear(X_pos.shape[1], args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
                         nn.Linear(args.hidden, 1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Negatives per step: balanced batches use a fresh random subset each step;
    # --neg_per_pos 0 uses every negative every step.
    if args.neg_per_pos > 0:
        m = min(len(Xn), max(args.min_neg_batch, args.neg_per_pos * len(Xp)))
    else:
        m = len(Xn)
    # Weight positives so both classes contribute equally to each step's loss.
    # Balanced batches keep this weight modest (about --neg_per_pos).
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(m / len(Xp), device=dev))
    y = torch.cat([torch.ones(len(Xp)), torch.zeros(m)]).to(dev)  # labels for one step: positives, then negatives

    head.train()
    for _ in range(args.steps):
        neg_idx = torch.randperm(len(Xn), device=dev)[:m]  # which negatives this step sees
        batch = torch.cat([Xp, Xn[neg_idx]])
        opt.zero_grad()
        loss_fn(head(batch).squeeze(1), y).backward()
        opt.step()
    head.eval()  # turns dropout off for scoring

    @torch.no_grad()
    def score(X_new):
        # Apply the same rescaling as in training, then return one score per image.
        return head((X_new - mu) / sd).squeeze(1).cpu().numpy()
    return score


def main():
    p = argparse.ArgumentParser(description="Validation vs test recall at fixed specificity, on cached features.")
    p.add_argument("--results_dir", type=str, required=True, help="Results folder (index.csv, labels.csv, dataset.json, <model>/)")
    p.add_argument("--feature_model", type=str, required=True, choices=list(models.MODELS), help="Cached features to train on")
    p.add_argument("--concepts", type=str, default="", help="Optional comma-separated subset of concepts; default is all categories")
    p.add_argument("--sizes", type=str, default="10,20,50,100,200,500", help="Comma-separated numbers of labeled positives")
    p.add_argument("--draws", type=int, default=30, help="Random draws per concept and size")
    p.add_argument("--val_frac", type=float, default=0.2, help="Fraction of the drawn positives and negatives held out for validation")
    p.add_argument("--n_neg", type=int, default=5000, help="Negatives per draw, split into training and validation like the positives")
    p.add_argument("--neg_per_pos", type=int, default=10, help="Negatives per positive in each training step; 0 = all negatives every step")
    p.add_argument("--min_neg_batch", type=int, default=64, help="Minimum negatives per training step (balanced batches only)")
    p.add_argument("--spec", type=float, default=0.9, help="Target specificity for the threshold")
    p.add_argument("--hidden", type=int, default=256, help="Hidden units in the head")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout after the hidden layer")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay")
    p.add_argument("--steps", type=int, default=300, help="Training steps per head")
    p.add_argument("--test_split", type=str, default="val2017", help="Split always placed in the test set")
    p.add_argument("--extra_test", type=int, default=20000, help="Additional images moved from other splits into the test set")
    p.add_argument("--split_seed", type=int, default=0, help="Seed for choosing the extra test images")
    p.add_argument("--min_fraction", type=float, default=0.0, help="Minimum pixel fraction to count as present; smaller instances are excluded")
    p.add_argument("--shard", type=int, default=0, help="This process's shard index (0-based)")
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards; concepts are interleaved across them")
    p.add_argument("--out_tag", type=str, default="", help="Optional tag appended to the output file name")
    args = p.parse_args()

    # ---- Load the dataset description, image index, embeddings, and labels ----
    with open(args.results_dir + "/dataset.json") as f:
        meta = json.load(f)
    # assert meta["exhaustive"], "recall_error.py currently requires exhaustive labels"
    categories = meta["categories"]
    with open(args.results_dir + "/index.csv") as f:
        index = list(csv.DictReader(f))

    model_dir = args.results_dir + "/" + args.feature_model
    emb_np = models.load_embeddings(model_dir)  # one embedding per image, in index.csv order
    assert emb_np.shape[0] == len(index), f"{emb_np.shape[0]} embeddings for {len(index)} images: finish embed.py first"
    Y = label_matrix(args.results_dir, categories, len(index), args.min_fraction, meta["exhaustive"])
    test = make_test_mask(index, args.test_split, args.extra_test, args.split_seed)
    emb = torch.from_numpy(emb_np).to("cuda")  # GPU chosen by CUDA_VISIBLE_DEVICES; all embeddings loaded once    
    print(f"test images: {test.sum()}, training pool: {(~test).sum()}")

    def rows_to_gpu(rows):
        """Embeddings for a list of image row numbers."""
        return emb[torch.from_numpy(rows).to(emb.device)]

    # ---- Decide which concepts this process handles ----
    sizes = [int(s) for s in args.sizes.split(",")]
    selected = set(args.concepts.split(",")) if args.concepts else None
    concepts = [(c, name) for c, name in enumerate(categories) if not selected or name in selected]
    concepts = concepts[args.shard::args.num_shards]  # e.g. shard 0 takes concepts 0, 2, 4, ...; shard 1 takes 1, 3, 5, ...

    rows = []
    for c, concept in concepts:
        t0 = time.time()
        y = Y[:, c]  # this concept's label for every image

        # Test-set images for this concept (excluded -1 images are left out entirely).
        test_c = test & (y >= 0)
        X_test_pos = rows_to_gpu(np.where(test_c & (y == 1))[0])
        X_test_neg = rows_to_gpu(np.where(test_c & (y == 0))[0])

        # Training pool: everything not in the test set, split into positives and negatives.
        pool_pos = np.where(~test & (y == 1))[0]
        pool_neg = np.where(~test & (y == 0))[0]

        for draw in range(args.draws):
            # Each draw simulates one practitioner collecting their own labeled data.
            rng = np.random.default_rng([draw, c])  # seed depends on draw and concept: reproducible
            perm = rng.permutation(pool_pos)  # random order of positives; each size takes the first N

            # One fixed split of this draw's negatives into validation and training.
            negs = rng.choice(pool_neg, size=args.n_neg, replace=False)
            n_val_neg = int(round(args.n_neg * args.val_frac))
            val_neg, train_neg = negs[:n_val_neg], negs[n_val_neg:]
            X_val_neg, X_train_neg = rows_to_gpu(val_neg), rows_to_gpu(train_neg)

            for n in sizes:
                if n > len(perm):  # not enough positives in the pool for this size
                    continue
                pos = perm[:n]  # the N labeled positives for this draw
                n_val = max(1, int(round(n * args.val_frac)))
                val_pos, train_pos = pos[:n_val], pos[n_val:]  # one split of the N positives

                # Train on the training split only.
                score = train_head(rows_to_gpu(train_pos), X_train_neg, args, seed=draw * 100003 + n * 101)

                # Threshold: the score below which --spec (0.9) of the validation negatives fall.
                thr = np.quantile(score(X_val_neg), args.spec)

                # Measured recall: validation positives above the threshold.
                val_recall = float((score(rows_to_gpu(val_pos)) > thr).mean())
                # True recall: the same head and threshold on the large test set.
                test_recall = float((score(X_test_pos) > thr).mean())
                test_spec = float((score(X_test_neg) <= thr).mean())  # check: should be close to --spec

                rows.append([concept, n, draw, n_val, val_recall, test_recall, val_recall - test_recall,
                             test_spec, len(X_test_pos)])
        print(f"{concept}: {len(pool_pos)} pool positives, {len(X_test_pos)} test positives, "
              f"{time.time() - t0:.0f}s", flush=True)

    # ---- Write one row per (concept, N, draw) ----
    out_file = model_dir + "/recall_error" + ("_" + args.out_tag if args.out_tag else "") + ".csv"
    with open(out_file, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["concept", "n_pos", "draw", "n_val_pos", "val_recall", "test_recall", "error",
                     "test_spec", "n_test_pos"])
        for r in rows:
            wr.writerow([round(v, 5) if isinstance(v, float) else v for v in r])
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()