"""
Choose the DNN thresholds on the validation set.

  python select_thresholds.py <output_dir>/val_predictions.csv

Uses the same rule as compare_logs.py:
  an image is changed    if its DNN score >= image threshold
  a cluster is changed   if the fraction of its images that are changed >= cluster threshold
Every (image threshold, cluster threshold) pair on the grid is scored on all val clusters (all label sources),
and the pair with the highest HM of sensitivity and specificity is reported (if several tie, the middle
of the tied region).

Grid
  image threshold     81 values, log-spaced from 0.0001 to 1
  cluster threshold   0.05 to 0.95 in steps of 0.05

Output
  printed: the best pair, its val TP/TN/FP/FN, and the 10 best pairs
  <same directory>/threshold_grid.csv: one row per pair with TP/TN/FP/FN and HM (for the ROC)
  <same directory>/thresholds.json:    the chosen pair, read by run_filters.py together with the model
Pass the chosen pair to compare_all.sh for the test set.
"""
import argparse
import csv
import json
import os
import numpy as np

IMAGE_THRESHOLDS = np.logspace(-4, 0, 81)
CLUSTER_THRESHOLDS = np.round(np.arange(1, 20) * 0.05, 2)


def load_clusters(path):
    """{(run, source_file, cluster_id): (label, sorted image scores)}"""
    clusters = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            c = clusters.setdefault((r['run'], r['source_file'], r['cluster_id']), [int(r['label']), []])
            c[1].append(float(r['score']))
    return {k: (label, np.sort(scores)) for k, (label, scores) in clusters.items()}


def hm(tp, tn, fp, fn):
    """Harmonic mean of sensitivity (TP rate) and specificity (TN rate)."""
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return 2 * sens * spec / (sens + spec) if sens + spec else 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose DNN image and cluster thresholds on the validation predictions')
    parser.add_argument('val_predictions', type=str, help='val_predictions.csv written by train_change_filter.py')
    parser.add_argument('--grid_csv', type=str, default='threshold_grid.csv', help='grid results file, written next to val_predictions (default=threshold_grid.csv)')
    args = parser.parse_args()

    clusters = load_clusters(args.val_predictions)
    labels = np.array([label for label, _ in clusters.values()])
    print(f"{len(clusters)} val clusters ({labels.sum()} changed), {sum(len(s) for _, s in clusters.values())} images")

    # fraction of each cluster's images with score >= t, for every image threshold t: shape (clusters, thresholds)
    frac = np.array([(len(s) - np.searchsorted(s, IMAGE_THRESHOLDS, side='left')) / len(s) for _, s in clusters.values()])

    results = []
    for i, t_img in enumerate(IMAGE_THRESHOLDS):
        for t_cl in CLUSTER_THRESHOLDS:
            pred = frac[:, i] >= t_cl
            tp = int((pred & (labels == 1)).sum())
            tn = int((~pred & (labels == 0)).sum())
            fp = int((pred & (labels == 0)).sum())
            fn = int((~pred & (labels == 1)).sum())
            results.append((float(t_img), float(t_cl), tp, tn, fp, fn, hm(tp, tn, fp, fn)))

    grid_path = os.path.dirname(os.path.abspath(args.val_predictions)) + "/" + args.grid_csv
    with open(grid_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_threshold', 'cluster_threshold', 'tp', 'tn', 'fp', 'fn', 'hm'])
        for t_img, t_cl, tp, tn, fp, fn, h in results:
            w.writerow([f"{t_img:.6g}", f"{t_cl:.2f}", tp, tn, fp, fn, f"{h:.4f}"])

    ranked = sorted(results, key=lambda r: -r[6])
    print(f"\n{'image_thr':>10s} {'cluster_thr':>11s} {'TP':>5s} {'TN':>5s} {'FP':>5s} {'FN':>5s} {'HM':>6s}")
    for t_img, t_cl, tp, tn, fp, fn, h in ranked[:10]:
        print(f"{t_img:10.4g} {t_cl:11.2f} {tp:5d} {tn:5d} {fp:5d} {fn:5d} {h:6.3f}")
    # if several pairs tie for the best HM, take the middle of the tied region rather than its edge:
    # the median image threshold (on a log scale), then the cluster threshold closest to 0.5
    tied = [r for r in results if abs(r[6] - ranked[0][6]) < 1e-12]
    mid = np.median(np.log10([r[0] for r in tied]))
    best = min(tied, key=lambda r: (abs(np.log10(r[0]) - mid), abs(r[1] - 0.5)))
    print(f"\nbest: image threshold {best[0]:.4g}, cluster threshold {best[1]:.2f}, "
          f"TP {best[2]} TN {best[3]} FP {best[4]} FN {best[5]}, HM {best[6]:.3f}")
    if len(tied) > 1:
        print(f"      ({len(tied)} pairs tie at this HM; this is the middle of the tied region)")
    print(f"grid written to {grid_path}")

    # the chosen pair, next to the model's other files, for run_filters.py to load with the model
    thresholds_path = os.path.dirname(os.path.abspath(args.val_predictions)) + "/thresholds.json"
    with open(thresholds_path, 'w') as f:
        json.dump({'image_threshold': best[0], 'cluster_threshold': best[1],
                   'val': {'tp': best[2], 'tn': best[3], 'fp': best[4], 'fn': best[5], 'hm': best[6]},
                   'selected_from': os.path.abspath(args.val_predictions), 'criterion': 'max HM of sensitivity and specificity'},
                  f, indent=1)
    print(f"thresholds written to {thresholds_path}")