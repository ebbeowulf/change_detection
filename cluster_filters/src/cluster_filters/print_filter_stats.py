"""
Print one filter's results as CSV: one line per prompt with TP, TN, FP, FN, precision, recall and F1,
summed over every labeled run below root_dir, plus a final 'all' line. Nothing is written to disk.

  python print_filter_stats.py ROOT_DIR FILTER_NAME [CONFIG_NAME]

  ROOT_DIR      searched for <run>/change2.0_results/<CONFIG_NAME>/filters/<FILTER_NAME>/results.json
  FILTER_NAME   the filter output folder, e.g. vlm_change__qwen3.5_35b+nothink+json
  CONFIG_NAME   step-3 configuration folder (default: sam3_0.1_openVocab)

Clusters are matched by ID with the labels in their configuration folder (<prompt>.labels.json); a cluster
still active after the filter counts as predicted changed. Runs without labels and unlabeled clusters are
not counted (reported on stderr, so stdout stays plain CSV).
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys


def load_labels(config_dir):
    labels = {}
    for path in glob.glob(config_dir + "/*.labels.json"):
        with open(path) as f:
            labels.update(json.load(f))
    return labels


def row(name, c):
    tp, tn, fp, fn = c['tp'], c['tn'], c['fp'], c['fn']
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [name, tp, tn, fp, fn, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print one filter's results per prompt as CSV")
    parser.add_argument('root_dir', type=str, help='folder to search, e.g. /data3/datasets/smart_change/s120')
    parser.add_argument('filter_name', type=str, help='filter output folder, e.g. vlm_change__qwen3.5_35b+nothink+json')
    parser.add_argument('config_name', type=str, nargs='?', default='sam3_0.1_openVocab', help='step-3 configuration folder (default=sam3_0.1_openVocab)')
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir.rstrip('/'))
    paths = sorted(glob.glob(root + "/**/change2.0_results/" + args.config_name + "/filters/" + args.filter_name
                             + "/results.json", recursive=True))
    if not paths:
        raise SystemExit(f"no {args.filter_name} results for {args.config_name} under {root}")

    counts = collections.defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})
    runs, runs_without_labels, unlabeled = 0, 0, 0
    for path in paths:
        config_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
        labels = load_labels(config_dir)
        if not labels:
            runs_without_labels += 1
            continue
        runs += 1
        with open(path) as f:
            clusters = json.load(f)['clusters']
        for uid, c in clusters.items():
            if uid not in labels:
                unlabeled += 1
                continue
            changed = bool(labels[uid].get('changed'))
            kept = c['status'] == 'active'
            key = ('t' if changed == kept else 'f') + ('p' if kept else 'n')
            counts[c['prompt']][key] += 1

    total = collections.Counter()
    w = csv.writer(sys.stdout)
    w.writerow(['prompt', 'TP', 'TN', 'FP', 'FN', 'precision', 'recall', 'F1'])
    for prompt in sorted(counts):
        w.writerow(row(prompt, counts[prompt]))
        total.update(counts[prompt])
    w.writerow(row('all', {k: total[k] for k in ('tp', 'tn', 'fp', 'fn')}))
    print(f"{runs} labeled runs counted, {runs_without_labels} runs without labels skipped, "
          f"{unlabeled} unlabeled clusters skipped", file=sys.stderr)