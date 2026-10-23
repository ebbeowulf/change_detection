"""
Train one of three change filters, then write raw per-image scores for compare_logs.py.

Models (--model)
  resnet    ImageNet ResNet-18 fine-tuned on the 1-channel delta crop around each box
  siamese   small head on frozen DINOv2 CLS embeddings of the before (render) and after crops
  patch     small head comparing frozen DINOv2 patch tokens before vs after, pooled toward high-delta patches

Training
  Train/val come from the split file (see make_holdout_split.py). Every sample is an image of a labeled
  cluster (the cluster's all_images frames, from build_dataset.py); no image is filtered out. The kept
  checkpoint has the best val image ROC-AUC.

Output (in output_dir)
  best.pt              model weights
  train_log.json       settings and per-epoch val numbers
  predictions.csv      one row per test image: image_index, environment, run, source_file, prompt, cluster_id,
                       frame_id, label, score (raw probability, no threshold applied)
  val_predictions.csv  the same for every val image (--write_val 1)

  --predict_only 1 skips training: loads output_dir/best.pt and writes the prediction files
  (use the same --model and split as the training run).
"""
import argparse
import csv
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from delta_dataset import load_split, load_samples, DeltaCropDataset
from dino_features import add_image_paths, extract_features
from dino_heads import CachedTokenDataset, build_head

# per-model training defaults (used when the matching argument is left at -1)
DEFAULTS = {'resnet':  {'epochs': 20, 'batch_size': 32, 'lr': 1e-4, 'weight_decay': 1e-4},
            'siamese': {'epochs': 30, 'batch_size': 64, 'lr': 1e-3, 'weight_decay': 1e-2},
            'patch':   {'epochs': 30, 'batch_size': 64, 'lr': 1e-3, 'weight_decay': 1e-2}}


# ---------------------------------------------------------------- models

class ResNetFilter(nn.Module):
    """ImageNet ResNet-18 with a 1-channel first conv (pretrained RGB filters summed) and one output logit."""

    def __init__(self):
        super().__init__()
        self.net = torchvision.models.resnet18(weights='DEFAULT')
        old = self.net.conv1
        self.net.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                                   stride=old.stride, padding=old.padding, bias=False)
        self.net.conv1.weight.data = old.weight.data.sum(dim=1, keepdim=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_model(args, samples):
    """The network for --model. DINOv2 heads are sized from the cached features of the first sample."""
    if args.model == 'resnet':
        return ResNetFilter()
    before, _, delta, _ = CachedTokenDataset(samples)[0]
    return build_head(args.model, before.shape[-1], delta.shape[0])


def make_dataset(args, samples, train):
    """ResNet reads delta crops (augmented when training); DINOv2 heads read cached features."""
    if args.model == 'resnet':
        return DeltaCropDataset(samples, args.img_size, augment=train)
    return CachedTokenDataset(samples)


# ---------------------------------------------------------------- data

def load_split_samples(args, split, name, device):
    """Every image of every labeled cluster in one split (after the split's label-source filters),
    with image paths resolved and, for DINOv2 models, features cached."""
    f = split.get('filters', {}).get(name, {})
    include = f.get('include', args.include)
    exclude = args.exclude + f.get('exclude', [])
    samples, missing = load_samples(args.dataset_root, split[name], args.delta_model, include, exclude)
    n_all = len(samples)
    samples = add_image_paths(args.dataset_root, samples)   # same image set for every model
    n_pos = sum(s['label'] for s in samples)
    print(f"{name}: {n_all} images ({missing} without delta files), "
          f"{len(samples)} with color+render images ({n_pos} changed)")
    if args.model != 'resnet':
        extract_features(args.dataset_root, samples, args.dino_model, args.img_size, bool(args.gray),
                         args.extract_batch_size, device, args.extract_workers)
    return samples


# ---------------------------------------------------------------- training and prediction

def predict(model, loader, device):
    """Raw probability for every sample in loader order. Each batch is (inputs..., label)."""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = [t.to(device) for t in batch[:-1]]
            probs.append(torch.sigmoid(model(*inputs)).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def val_metrics(probs, labels):
    """Val progress: ROC-AUC (threshold-free, used to keep the best checkpoint) plus P/R/F1/specificity at 0.5."""
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    n_neg = int((labels == 0).sum())
    spec = float(((preds == 0) & (labels == 0)).sum() / n_neg) if n_neg else 0.0
    auc = float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else None
    return {'roc_auc': auc, 'precision': float(p), 'recall': float(r), 'f1': float(f1), 'specificity': spec}


def train(model, train_samples, val_samples, args, device):
    """Class-weighted BCE; saves best.pt whenever val image ROC-AUC improves."""
    n_pos = sum(s['label'] for s in train_samples)
    n_neg = len(train_samples) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training set needs both changed and unchanged samples")
    train_loader = DataLoader(make_dataset(args, train_samples, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(make_dataset(args, val_samples, False), batch_size=args.batch_size,
                            num_workers=args.num_workers)
    val_labels = np.array([s['label'] for s in val_samples])

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best_auc = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            inputs = [t.to(device) for t in batch[:-1]]
            y = batch[-1].to(device)
            optimizer.zero_grad()
            loss = criterion(model(*inputs), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        m = val_metrics(predict(model, val_loader, device), val_labels)
        history.append({'epoch': epoch + 1, 'train_loss': total / len(train_samples), 'val': m})
        print(f"epoch {epoch + 1:3d}  loss {total / len(train_samples):.4f}  val images: ROC-AUC {m['roc_auc']}  "
              f"(at 0.5: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} Spec {m['specificity']:.3f})")
        if m['roc_auc'] is not None and m['roc_auc'] > best_auc:
            best_auc = m['roc_auc']
            torch.save(model.state_dict(), args.output_dir + "/best.pt")
    with open(args.output_dir + "/train_log.json", 'w') as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)


def write_predictions(model, samples, csv_path, args, device):
    """Raw score for every image (no threshold applied)."""
    loader = DataLoader(make_dataset(args, samples, False), batch_size=args.batch_size, num_workers=args.num_workers)
    probs = predict(model, loader, device)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_index', 'environment', 'run', 'source_file', 'prompt', 'cluster_id', 'frame_id', 'label', 'score'])
        for i, (p, s) in enumerate(zip(probs, samples)):
            w.writerow([i, s['environment'], s['run'], s['source_file'], s['prompt'], s['cluster_id'], s['frame_id'],
                        s['label'], f"{p:.6f}"])
    print(f"wrote {len(samples)} images to {csv_path}")


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description='Train a change filter (resnet, siamese or patch) and write raw per-image scores')
    parser.add_argument('dataset_root', type=str, help='root of the change dataset (e.g. /data2/datasets/change_dataset)')
    parser.add_argument('split_file', type=str, help='split file name inside <dataset_root>/splits (e.g. holdout_s120.json)')
    parser.add_argument('output_dir', type=str, help='directory for best.pt and the prediction files')
    parser.add_argument('--model', type=str, required=True, help="'resnet' (delta crops), 'siamese' (DINOv2 CLS) or 'patch' (DINOv2 delta-masked patches)")
    parser.add_argument('--predict_only', type=int, default=0, help='1 = skip training, load output_dir/best.pt and write prediction files (default=0)')
    parser.add_argument('--write_val', type=int, default=1, help='1 = also write val_predictions.csv for every val image (default=1)')
    parser.add_argument('--delta_model', type=str, default='sam3', help='which delta_change/<model> masks to use (default=sam3)')
    parser.add_argument('--include', type=str, nargs='*', default=[], help='train/val: only label-source files whose name contains one of these')
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='skip label-source files whose name contains one of these')
    parser.add_argument('--img_size', type=int, default=224, help='crop size; multiple of 14 for DINOv2 (default=224)')
    parser.add_argument('--dino_model', type=str, default='facebook/dinov2-small', help='siamese/patch: HuggingFace DINOv2 model (default=facebook/dinov2-small)')
    parser.add_argument('--gray', type=int, default=1, help='siamese/patch: 1 = grayscale crops copied to 3 channels, 0 = RGB (default=1)')
    parser.add_argument('--extract_batch_size', type=int, default=128, help='siamese/patch: samples per DINOv2 forward pass when building the cache (default=128)')
    parser.add_argument('--extract_workers', type=int, default=16, help='siamese/patch: worker processes decoding/cropping images for the cache (default=16)')
    parser.add_argument('--epochs', type=int, default=-1, help='training epochs (default=-1: resnet 20, siamese/patch 30)')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size (default=-1: resnet 32, siamese/patch 64)')
    parser.add_argument('--lr', type=float, default=-1, help='AdamW learning rate (default=-1: resnet 1e-4, siamese/patch 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=-1, help='AdamW weight decay (default=-1: resnet 1e-4, siamese/patch 1e-2)')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader worker processes (default=4)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default=0)')
    args = parser.parse_args()
    if args.model not in DEFAULTS:
        parser.error(f"--model must be one of {list(DEFAULTS)}")
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) < 0:
            setattr(args, k, v)
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    split = load_split(args.dataset_root, args.split_file)

    test = load_split_samples(args, split, 'test', device)
    if not test:
        raise SystemExit("test split is empty")
    model = build_model(args, test).to(device)       # DINOv2 heads are sized from the test features

    val = load_split_samples(args, split, 'val', device) if (args.write_val or not args.predict_only) else []
    if not args.predict_only:
        train(model, load_split_samples(args, split, 'train', device), val, args, device)

    model.load_state_dict(torch.load(args.output_dir + "/best.pt", map_location=device))
    write_predictions(model, test, args.output_dir + "/predictions.csv", args, device)
    if args.write_val:
        write_predictions(model, val, args.output_dir + "/val_predictions.csv", args, device)