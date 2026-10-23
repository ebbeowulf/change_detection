import argparse
import csv
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from delta_dataset import load_split, load_samples, add_box_fractions, DeltaCropDataset
from resnet_model import build_resnet


def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0)


def binary_metrics(preds, labels):
    out = {'n': int(len(labels)), 'n_pos': int(labels.sum())}
    if len(labels) == 0:
        return out
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    # specificity = TNR; hmean = harmonic mean of sensitivity (TPR = recall) and specificity
    n_neg = int((labels == 0).sum())
    spec = float(((preds == 0) & (labels == 0)).sum() / n_neg) if n_neg else 0.0
    hmean = 2 * r * spec / (r + spec) if (r + spec) > 0 else 0.0
    out.update({'precision': float(p), 'recall': float(r), 'f1': float(f1), 'specificity': spec,
                'hmean': float(hmean), 'accuracy': float((preds == labels).mean())})
    return out


def image_metrics(probs, labels, threshold):
    out = binary_metrics((probs >= threshold).astype(int), labels)
    # ROC-AUC is undefined without both classes present
    out['roc_auc'] = float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else None
    return out


def cluster_key(s):
    return (s['run'], s['source_file'], s['cluster_id'])


def cluster_votes(samples, probs, all_clusters, threshold, tie):
    """Majority vote per cluster (run + label-source file + cluster ID) over the threshold-filtered images.
    all_clusters maps every cluster in the unfiltered set to (label, image count), so a cluster whose
    images were all filtered out is still evaluated, as predicted not changed. Ties follow the tie rule."""
    groups = {key: {'label': label, 'votes': [], 'n_images': n} for key, (label, n) in all_clusters.items()}
    for s, prob in zip(samples, probs):
        groups[cluster_key(s)]['votes'].append(int(prob >= threshold))
    rows = []
    for (run, source_file, cluster_id), g in sorted(groups.items()):
        n, yes = len(g['votes']), sum(g['votes'])
        if n == 0:
            pred = 0
        else:
            pred = int(2 * yes > n or (2 * yes == n and tie == 'changed'))
        rows.append({'run': run, 'source_file': source_file, 'cluster_id': cluster_id, 'label': g['label'],
                     'pred': pred, 'votes_changed': yes, 'votes_counted': n, 'n_images': g['n_images']})
    return rows


def evaluate(samples, probs, all_clusters, args):
    labels = np.array([s['label'] for s in samples], dtype=int)
    rows = cluster_votes(samples, probs, all_clusters, args.threshold, args.tie)
    cluster = binary_metrics(np.array([r['pred'] for r in rows], dtype=int), np.array([r['label'] for r in rows], dtype=int))
    cluster['n_no_valid_images'] = sum(r['votes_counted'] == 0 for r in rows)
    return {'image': image_metrics(probs, labels, args.threshold), 'cluster': cluster}, rows


def describe(name, samples, missing):
    n_pos = sum(s['label'] for s in samples)
    print(f"{name}: {len(samples)} samples ({n_pos} changed, {len(samples) - n_pos} not), {missing} skipped for missing deltas")


def fmt(m):
    return (f"P {m.get('precision', 0):.3f} R {m.get('recall', 0):.3f} F1 {m.get('f1', 0):.3f} "
            f"Spec {m.get('specificity', 0):.3f} HM {m.get('hmean', 0):.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet binary change classifier on delta-mask crops')
    parser.add_argument('dataset_root', type=str, help='root of the change dataset (e.g. /data2/datasets/change_dataset)')
    parser.add_argument('split_file', type=str, help='split file name inside <dataset_root>/splits (e.g. phone_runs.json)')
    parser.add_argument('output_dir', type=str, help='directory for checkpoints, metrics, and test predictions')
    parser.add_argument('--delta_model', type=str, default='sam3', help='which delta_change/<model> masks to use as input (default=sam3)')
    parser.add_argument('--include', type=str, nargs='*', default=[], help="only use label-source files whose name contains one of these (e.g. openVocab save_results-orig)")
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='skip label-source files whose name contains one of these')
    parser.add_argument('--delta_threshold', type=float, default=0.3, help='delta value a box pixel must exceed to count toward its fraction (default=0.3)')
    parser.add_argument('--train_min_frac', type=float, default=0.0, help='drop training images whose box has less than this fraction of pixels above --delta_threshold (default=0.0 = keep all)')
    parser.add_argument('--eval_min_frac', type=float, default=0.0, help='drop val/test images whose box has less than this fraction of pixels above --delta_threshold; remaining images are scored and vote for their cluster (default=0.0 = keep all)')
    parser.add_argument('--tie', type=str, default='changed', help="cluster vote tie rule: 'changed' or 'not_changed' (default=changed)")
    parser.add_argument('--arch', type=str, default='resnet18', help='torchvision resnet variant (default=resnet18)')
    parser.add_argument('--pretrained', type=int, default=1, help='1 = start from ImageNet weights (default=1)')
    parser.add_argument('--img_size', type=int, default=224, help='crop size fed to the network (default=224)')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs (default=20)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size (default=32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for AdamW (default=1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW weight decay (default=1e-4)')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold for an image to vote changed (default=0.5)')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader worker processes (default=4)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default=0)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    split = load_split(args.dataset_root, args.split_file)
    sets = {}
    all_clusters = {'val': {}, 'test': {}}
    for name in ('train', 'val', 'test'):
        # a split file may restrict one split to specific label sources (see make_holdout_split.py);
        # its include replaces --include for that split, and its exclude adds to --exclude
        f = split.get('filters', {}).get(name, {})
        include = f.get('include', args.include)
        exclude = args.exclude + f.get('exclude', [])
        if f:
            print(f"{name} label sources: include {include} exclude {exclude}")
        samples, missing = load_samples(args.dataset_root, split[name], args.delta_model, include, exclude)
        add_box_fractions(args.dataset_root, samples, args.delta_model, args.delta_threshold)
        describe(name, samples, missing)
        if name != 'train':
            # record every cluster before filtering so fully-obstructed clusters still count in cluster metrics
            for s in samples:
                label, n = all_clusters[name].get(cluster_key(s), (s['label'], 0))
                all_clusters[name][cluster_key(s)] = (label, n + 1)
        min_frac = args.train_min_frac if name == 'train' else args.eval_min_frac
        if min_frac > 0:
            n_before = len(samples)
            samples = [s for s in samples if s['frac_above'] >= min_frac]
            n_pos = sum(s['label'] for s in samples)
            print(f"  {name} after frac >= {min_frac}: {len(samples)} samples ({n_pos} changed, "
                  f"{len(samples) - n_pos} not), {n_before - len(samples)} removed")
        sets[name] = samples
    n_pos = sum(s['label'] for s in sets['train'])
    n_neg = len(sets['train']) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training set needs both changed and unchanged samples")

    loaders = {name: DataLoader(DeltaCropDataset(sets[name], args.img_size, augment=(name == 'train')),
                                batch_size=args.batch_size, shuffle=(name == 'train'), num_workers=args.num_workers)
               for name in sets}

    model = build_resnet(in_channels=1, arch=args.arch, pretrained=bool(args.pretrained)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        train_loss = total / len(sets['train'])

        val, _ = evaluate(sets['val'], predict(model, loaders['val'], device), all_clusters['val'], args)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val': val})
        print(f"epoch {epoch + 1:3d}  loss {train_loss:.4f}  val image {fmt(val['image'])} "
              f"ROC-AUC {val['image'].get('roc_auc')}  |  cluster {fmt(val['cluster'])}")

        # keep the checkpoint with the best val image ROC-AUC (F1 if ROC-AUC is undefined)
        m = val['image']
        score = m['roc_auc'] if m.get('roc_auc') is not None else m.get('f1', 0.0)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.output_dir + "/best.pt")

    model.load_state_dict(torch.load(args.output_dir + "/best.pt", map_location=device))
    probs = predict(model, loaders['test'], device)
    test, rows = evaluate(sets['test'], probs, all_clusters['test'], args)
    print("TEST (best val checkpoint)")
    for k in ('image', 'cluster'):
        print(f"  {k:15s} {test[k]}")

    with open(args.output_dir + "/metrics.json", 'w') as f:
        json.dump({'args': vars(args), 'history': history, 'test': test}, f, indent=2)
    with open(args.output_dir + "/test_predictions.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prob', 'label', 'frac_above', 'environment', 'run', 'source_file', 'cluster_id', 'frame_id'])
        for prob, s in zip(probs, sets['test']):
            writer.writerow([f"{prob:.4f}", s['label'], f"{s['frac_above']:.4f}", s['environment'], s['run'],
                             s['source_file'], s['cluster_id'], s['frame_id']])
    with open(args.output_dir + "/test_clusters.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['run'])
        writer.writeheader()
        writer.writerows(rows)