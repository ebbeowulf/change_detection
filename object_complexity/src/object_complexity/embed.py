#!/usr/bin/env python3
"""Stage 2: embed every image in index.csv with one model.

Writes to results_dir/<model>/:
  emb_00000.npy, emb_00001.npy, ...  float16 (shard_size, dim), rows in index.csv order
  meta.json                          checkpoint, shard size, embedding dim, logit scale/bias
Shards that already exist are skipped, so an interrupted run can be resumed.
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import models


class ImageSet(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return models.preprocess_image(self.processor, img)


def main():
    p = argparse.ArgumentParser(description="Embed all indexed images with one model.")
    p.add_argument("--results_dir", type=str, required=True, help="Results folder containing index.csv")
    p.add_argument("--model", type=str, required=True, choices=list(models.MODELS), help="Model name from models.py")
    p.add_argument("--index_file", type=str, default="index.csv", help="Index file name in results_dir")
    p.add_argument("--shard_size", type=int, default=10000, help="Images per embedding shard")
    p.add_argument("--batch_size", type=int, default=64, help="Images per GPU batch")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader worker processes for image loading")
    p.add_argument("--device", type=str, default="cuda", help="Torch device")
    args = p.parse_args()

    with open(args.results_dir + "/" + args.index_file) as f:
        rows = list(csv.DictReader(f))
    assert all(int(r["row"]) == i for i, r in enumerate(rows)), "index rows must be 0..N-1 in order"
    paths = [r["path"] for r in rows]

    out_dir = args.results_dir + "/" + args.model
    os.makedirs(out_dir, exist_ok=True)
    meta_file = out_dir + "/meta.json"
    checkpoint = models.MODELS[args.model]["checkpoint"]
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        assert meta["shard_size"] == args.shard_size, "shard_size differs from existing shards"
        assert meta["checkpoint"] == checkpoint, "checkpoint differs from existing shards"
    else:
        meta = {"model": args.model, "checkpoint": checkpoint, "shard_size": args.shard_size}

    model, processor = models.load(args.model, args.device)
    meta["logit_scale"], meta["logit_bias"] = models.score_params(model)
    meta["n_images"] = len(paths)
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    n_shards = (len(paths) + args.shard_size - 1) // args.shard_size
    for s in range(n_shards):
        shard_file = out_dir + f"/emb_{s:05d}.npy"
        if os.path.exists(shard_file):
            continue
        lo, hi = s * args.shard_size, min((s + 1) * args.shard_size, len(paths))
        loader = DataLoader(ImageSet(paths[lo:hi], processor), batch_size=args.batch_size,
                            num_workers=args.num_workers)
        feats = [models.encode_pixels(model, px.to(args.device)).cpu().numpy() for px in loader]
        emb = np.concatenate(feats).astype(np.float16)
        tmp_file = shard_file + ".tmp"
        with open(tmp_file, "wb") as f:
            np.save(f, emb)
        os.replace(tmp_file, shard_file)  # only complete shards get the final name
        print(f"shard {s + 1}/{n_shards}: rows {lo}-{hi - 1}, dim {emb.shape[1]}")
        meta["dim"] = int(emb.shape[1])
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()