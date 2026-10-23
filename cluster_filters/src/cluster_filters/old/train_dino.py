"""
Train a classifier head on frozen DINOv2 features of before/after crops, then write raw test scores.

Training
  Train/val come from the split file. Images below --train_min_frac (train) or --eval_min_frac (val)
  are dropped; the best checkpoint is the one with the highest val image ROC-AUC.

Output (in output_dir)
  best.pt              the head's weights
  predictions.csv      one row per test image, for EVERY image of every test cluster (no visibility filter,
                       no threshold, no voting): image_index, environment, run, source_file, prompt,
                       cluster_id, frame_id, label, score (raw probability)
  box_counts.npz       for the same images, by image_index: grid (delta thresholds), areas (box pixels),
                       counts (box pixels with delta > each grid value). Lets compare_logs.py apply any
                       delta threshold with a pixel-count or fraction rule afterwards.
  val_predictions.csv / val_box_counts.npz   the same for every val image (--write_val 1), for learning
                       thresholds on val.
  All decisions are made later by compare_logs.py.

  --predict_only 1 skips training: loads output_dir/best.pt and writes the prediction files.
"""
import argparse
import csv
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from delta_dataset import load_split, load_samples, add_box_fractions, add_box_counts, DELTA_GRID
from dino_features import add_image_paths, extract_features
from dino_heads import CachedTokenDataset, build_head


def predict(head, loader, device):
    head.eval()
    probs = []
    with torch.no_grad():
        for b, a, d, _ in loader:
            probs.append(torch.sigmoid(head(b.to(device), a.to(device), d.to(device))).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def image_metrics(probs, labels, threshold=0.5):
    """Val progress only: P/R/F1/specificity at threshold, and ROC-AUC (threshold-free, used for checkpointing)."""
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    n_neg = int((labels == 0).sum())
    spec = float(((preds == 0) & (labels == 0)).sum() / n_neg) if n_neg else 0.0
    auc = float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else None
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1), 'specificity': spec, 'roc_auc': auc}


def load_split_samples(args, split, name, min_frac, device):
    """Samples for one split, after label-source filters and (if min_frac > 0) the visibility filter,
    with image paths resolved and DINOv2 features cached."""
    f = split.get('filters', {}).get(name, {})
    include = f.get('include', args.include)
    exclude = args.exclude + f.get('exclude', [])
    samples, missing = load_samples(args.dataset_root, split[name], args.delta_model, include, exclude)
    add_box_fractions(args.dataset_root, samples, args.delta_model, args.delta_threshold)
    n_all = len(samples)
    if min_frac > 0:
        samples = [s for s in samples if s['frac_above'] >= min_frac]
    n_kept = len(samples)
    samples = add_image_paths(args.dataset_root, samples)
    n_pos = sum(s['label'] for s in samples)
    print(f"{name}: {n_all} images ({missing} without delta files), {n_kept} after frac >= {min_frac}, "
          f"{len(samples)} with color+render images ({n_pos} changed)")
    extract_features(args.dataset_root, samples, args.dino_model, args.img_size, bool(args.gray),
                     args.extract_batch_size, device, args.extract_workers)
    return samples


def write_predictions(head, samples, csv_path, npz_path, args, device):
    """Raw score for every image, plus box area and pixel counts above each DELTA_GRID value."""
    add_box_counts(args.dataset_root, samples, args.delta_model)
    probs = predict(head, DataLoader(CachedTokenDataset(samples), batch_size=args.batch_size,
                                     num_workers=args.num_workers), device)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_index', 'environment', 'run', 'source_file', 'prompt', 'cluster_id', 'frame_id', 'label', 'score'])
        for i, (p, s) in enumerate(zip(probs, samples)):
            w.writerow([i, s['environment'], s['run'], s['source_file'], s['prompt'], s['cluster_id'], s['frame_id'],
                        s['label'], f"{p:.6f}"])
    np.savez(npz_path, grid=DELTA_GRID, areas=np.array([s['box_area'] for s in samples], dtype=np.int64),
             counts=np.stack([s['box_counts'] for s in samples]).astype(np.int32))
    print(f"wrote {len(samples)} images to {csv_path} and {npz_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a head on frozen DINOv2 features and write raw per-image test scores')
    parser.add_argument('dataset_root', type=str, help='root of the change dataset (e.g. /data2/datasets/change_dataset)')
    parser.add_argument('split_file', type=str, help='split file name inside <dataset_root>/splits (e.g. holdout_s120.json)')
    parser.add_argument('output_dir', type=str, help='directory for best.pt and predictions.csv')
    parser.add_argument('--predict_only', type=int, default=0, help='1 = skip training, load output_dir/best.pt and write predictions.csv (default=0)')
    parser.add_argument('--head', type=str, default='siamese', help="'siamese' (CLS comparison) or 'patch' (delta-masked patch comparison) (default=siamese)")
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-small', help='HuggingFace DINOv2 model (default=facebook/dinov2-small)')
    parser.add_argument('--gray', type=int, default=1, help='1 = grayscale crops copied to 3 channels, 0 = RGB (default=1)')
    parser.add_argument('--img_size', type=int, default=224, help='crop size, multiple of 14 (default=224)')
    parser.add_argument('--delta_model', type=str, default='sam3', help='which delta_change/<model> masks to use (default=sam3)')
    parser.add_argument('--include', type=str, nargs='*', default=[], help='train/val: only label-source files whose name contains one of these')
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='skip label-source files whose name contains one of these')
    parser.add_argument('--delta_threshold', type=float, default=0.1, help='training only: delta value a box pixel must exceed to count toward its fraction (default=0.1)')
    parser.add_argument('--train_min_frac', type=float, default=0.1, help='training only: drop training images below this fraction of box pixels above --delta_threshold (default=0.1)')
    parser.add_argument('--eval_min_frac', type=float, default=0.1, help='training only: drop val images below this fraction when choosing the checkpoint (default=0.1); prediction files are never filtered')
    parser.add_argument('--write_val', type=int, default=1, help='1 = also write val_predictions.csv / val_box_counts.npz for every val image (default=1)')
    parser.add_argument('--epochs', type=int, default=30, help='training epochs (default=30)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default=64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for AdamW (default=1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='AdamW weight decay (default=1e-2)')
    parser.add_argument('--extract_batch_size', type=int, default=128, help='samples per DINOv2 forward pass when building the cache (default=128)')
    parser.add_argument('--extract_workers', type=int, default=16, help='worker processes decoding/cropping images while building the cache (default=16)')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader worker processes (default=4)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default=0)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    split = load_split(args.dataset_root, args.split_file)

    test = load_split_samples(args, split, 'test', 0.0, device)   # every image, unfiltered
    if not test:
        raise SystemExit("test split is empty")
    b0, _, d0, _ = CachedTokenDataset(test)[0]
    head = build_head(args.head, b0.shape[-1], d0.shape[0]).to(device)

    if not args.predict_only:
        train = load_split_samples(args, split, 'train', args.train_min_frac, device)
        val = load_split_samples(args, split, 'val', args.eval_min_frac, device)
        n_pos = sum(s['label'] for s in train)
        n_neg = len(train) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError("training set needs both changed and unchanged samples")
        train_loader = DataLoader(CachedTokenDataset(train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(CachedTokenDataset(val), batch_size=args.batch_size, num_workers=args.num_workers)
        val_labels = np.array([s['label'] for s in val])

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        history, best_auc = [], -1.0
        for epoch in range(args.epochs):
            head.train()
            total = 0.0
            for b, a, d, y in train_loader:
                b, a, d, y = b.to(device), a.to(device), d.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(head(b, a, d), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
            m = image_metrics(predict(head, val_loader, device), val_labels)
            history.append({'epoch': epoch + 1, 'train_loss': total / len(train), 'val': m})
            print(f"epoch {epoch + 1:3d}  loss {total / len(train):.4f}  val images: ROC-AUC {m['roc_auc']}  "
                  f"(at 0.5: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} Spec {m['specificity']:.3f})")
            if m['roc_auc'] is not None and m['roc_auc'] > best_auc:
                best_auc = m['roc_auc']
                torch.save(head.state_dict(), args.output_dir + "/best.pt")
        with open(args.output_dir + "/train_log.json", 'w') as f:
            json.dump({'args': vars(args), 'history': history}, f, indent=2)

    head.load_state_dict(torch.load(args.output_dir + "/best.pt", map_location=device))
    write_predictions(head, test, args.output_dir + "/predictions.csv", args.output_dir + "/box_counts.npz", args, device)
    if args.write_val:
        val_all = load_split_samples(args, split, 'val', 0.0, device)   # every val image, unfiltered
        write_predictions(head, val_all, args.output_dir + "/val_predictions.csv",
                          args.output_dir + "/val_box_counts.npz", args, device)