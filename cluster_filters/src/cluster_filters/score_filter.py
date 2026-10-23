"""
Compare a filter sequence's results with the labels of the same step-3 configuration.

  python evaluate_filters.py <config_dir> <filter_name> [--target changed] [--output_csv results.csv]

  config_dir    step-3 configuration folder, e.g. <run>/change2.0_results/sam3_0.1_openVocab
  filter_name   folder under <config_dir>/filters/, i.e. the stage names joined by '+', e.g. siamese

Reads <config_dir>/filters/<filter_name>/results.json (every cluster, active or rejected) and every
<config_dir>/<prompt>.labels.json, matched by cluster ID. A cluster still active after the last stage
counts as predicted positive. Clusters without a label are counted separately, not as negatives.

Printed: TP/TN/FP/FN per prompt and in total, and for the false negatives, which stage rejected them.
--output_csv adds one row per prompt plus an 'all' row to a CSV (created if missing); rows for the same
config_dir / filter / target are replaced when rerun.
"""
import argparse
import collections
import csv
import glob
import json
import os

CSV_FIELDS = ['config_dir', 'filter', 'target', 'prompt', 'tp', 'tn', 'fp', 'fn', 'unlabeled', 'fn_rejected_by']


def load_labels(config_dir):
    labels = {}
    for path in glob.glob(config_dir + "/*.labels.json"):
        with open(path) as f:
            labels.update(json.load(f))
    return labels


def is_positive(label, target):
    if target == 'changed':
        return bool(label.get('changed'))
    return bool(label.get('changed')) and bool(label.get('pickup'))      # changed_and_pickup


def evaluate(clusters, labels, target):
    """Per-prompt confusion counts, unlabeled counts, and the stage that rejected each false negative."""
    counts = collections.defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})
    unlabeled = collections.Counter()
    lost_by = collections.defaultdict(collections.Counter)
    for uid, r in clusters.items():
        prompt = r['prompt']
        if uid not in labels:
            unlabeled[prompt] += 1
            continue
        truth = is_positive(labels[uid], target)
        kept = r['status'] == 'active'
        key = ('t' if truth == kept else 'f') + ('p' if kept else 'n')
        counts[prompt][key] += 1
        if key == 'fn':
            lost_by[prompt][r['rejected_by']] += 1
    return counts, unlabeled, lost_by


def write_csv(path, rows, config_dir, filter_name, target):
    """Replace earlier rows for the same config_dir / filter / target, then append these."""
    kept = []
    if os.path.exists(path):
        with open(path) as f:
            kept = [r for r in csv.DictReader(f)
                    if (r['config_dir'], r['filter'], r['target']) != (config_dir, filter_name, target)]
    tmp = path + ".tmp"
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(kept + rows)
    os.replace(tmp, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare a filter sequence's results with the labels")
    parser.add_argument('config_dir', type=str, help='step-3 configuration folder, e.g. <run>/change2.0_results/sam3_0.1_openVocab')
    parser.add_argument('filter_name', type=str, help="filter output folder under <config_dir>/filters/, e.g. siamese or image_filter+siamese")
    parser.add_argument('--target', type=str, default='changed', help="what counts as positive: 'changed' or 'changed_and_pickup' (default=changed)")
    parser.add_argument('--output_csv', type=str, default='', help='add the counts to this CSV (default = print only)')
    args = parser.parse_args()
    if args.target not in ('changed', 'changed_and_pickup'):
        parser.error("--target must be 'changed' or 'changed_and_pickup'")

    config_dir = os.path.abspath(args.config_dir.rstrip('/'))
    results_path = config_dir + "/filters/" + args.filter_name + "/results.json"
    if not os.path.exists(results_path):
        raise SystemExit(f"ERROR: {results_path} not found - run run_filters.py first")
    with open(results_path) as f:
        clusters = json.load(f)['clusters']
    labels = load_labels(config_dir)
    if not labels:
        raise SystemExit(f"ERROR: no *.labels.json files in {config_dir}")

    counts, unlabeled, lost_by = evaluate(clusters, labels, args.target)

    print(f"{config_dir}  filter: {args.filter_name}  target: {args.target}")
    print(f"  {'prompt':20s} {'TP':>4s} {'TN':>4s} {'FP':>4s} {'FN':>4s}  {'unlabeled':>9s}  FN rejected by")
    rows = []
    total, total_unlabeled, total_lost = collections.Counter(), 0, collections.Counter()
    for prompt in sorted(set(counts) | set(unlabeled)):
        c = counts[prompt]
        lost = dict(lost_by[prompt])
        print(f"  {prompt:20s} {c['tp']:4d} {c['tn']:4d} {c['fp']:4d} {c['fn']:4d}  {unlabeled[prompt]:9d}  {lost if lost else ''}")
        rows.append(dict(config_dir=config_dir, filter=args.filter_name, target=args.target, prompt=prompt,
                         unlabeled=unlabeled[prompt], fn_rejected_by=json.dumps(lost), **c))
        total.update(c)
        total_unlabeled += unlabeled[prompt]
        total_lost.update(lost_by[prompt])
    print(f"  {'all':20s} {total['tp']:4d} {total['tn']:4d} {total['fp']:4d} {total['fn']:4d}  "
          f"{total_unlabeled:9d}  {dict(total_lost) if total_lost else ''}")
    rows.append(dict(config_dir=config_dir, filter=args.filter_name, target=args.target, prompt='all',
                     tp=total['tp'], tn=total['tn'], fp=total['fp'], fn=total['fn'],
                     unlabeled=total_unlabeled, fn_rejected_by=json.dumps(dict(total_lost))))

    if args.output_csv:
        write_csv(args.output_csv, rows, config_dir, args.filter_name, args.target)
        print(f"counts written to {args.output_csv}")