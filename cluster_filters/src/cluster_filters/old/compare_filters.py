"""
Side-by-side comparison of the VLM filter and a trained model for ONE run and ONE prompt.

Inputs
  vlm_log       VLM "before_and_after_filter" log for that run/prompt.
                Uses its "scores": {cluster_id: fraction of is_new_object answers}.
                A cluster is changed if its score >= 0.5.
  clusters_csv  test_clusters.csv written by train_resnet.py / train_dino.py.
                Supplies the true label and the model's cluster prediction.
  run           the run as written in the csv's "run" column, or any unique part of it
                (e.g. s120/T1__changes__changes1, or T1__changes__changes1).
  prompt        the prompt (e.g. electronics).

Output (printed)
  One row per cluster: label, VLM score and decision, model votes and decision, which one was right.
  Then precision / recall / F1 / specificity / HM for each, and the clusters where they disagree.
  A cluster in the VLM log but not the model csv is an ERROR. A cluster in the model csv but not the
  VLM log is marked "model only". Both are left out of the side-by-side scores.
"""
import argparse
import csv
import json


def scores(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum(l and not p for p, l in zip(preds, labels))
    tn = sum(not p and not l for p, l in zip(preds, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    hm = 2 * recall * specificity / (recall + specificity) if recall + specificity else 0.0
    return precision, recall, f1, specificity, hm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the VLM log and a trained model for one run and prompt')
    parser.add_argument('vlm_log', type=str, help='VLM before_and_after_filter log (json) for this run/prompt')
    parser.add_argument('clusters_csv', type=str, help='test_clusters.csv from the trained model')
    parser.add_argument('run', type=str, help='run as in the csv "run" column, or a unique part of it')
    parser.add_argument('prompt', type=str, help='prompt, e.g. electronics')
    parser.add_argument('--vlm_threshold', type=float, default=0.5, help='VLM score at or above which a cluster is changed (default=0.5)')
    args = parser.parse_args()

    with open(args.vlm_log) as f:
        vlm = {str(k): float(v) for k, v in json.load(f)['scores'].items()}

    # model rows for this run and prompt
    with open(args.clusters_csv) as f:
        all_rows = list(csv.DictReader(f))
    rows = [r for r in all_rows if args.run in r['run'] and r['source_file'].endswith("__" + args.prompt + ".json")]
    runs = sorted({r['run'] for r in rows})
    sources = sorted({r['source_file'] for r in rows})
    if not rows:
        raise SystemExit(f"no rows in {args.clusters_csv} for run '{args.run}' and prompt '{args.prompt}'.\n"
                         f"runs in this csv: {sorted({r['run'] for r in all_rows})}")
    if len(runs) > 1 or len(sources) > 1:
        raise SystemExit(f"'{args.run}' / '{args.prompt}' matches more than one run or label source: {runs} {sources}")
    model = {r['cluster_id']: r for r in rows}
    print(f"run {runs[0]}   label source {sources[0]}\n")

    # every cluster either side knows about; the model csv carries the true label
    cids = sorted(set(vlm) | set(model), key=lambda c: int(c) if c.isdigit() else c)
    print(f"{'cluster':>7s}  {'label':>5s}  {'VLM score':>9s} {'VLM':>3s}  {'model votes':>11s} {'model':>5s}  right")
    labels, vlm_pred, model_pred, compared = [], [], [], []
    for c in cids:
        v = vlm.get(c)
        m = model.get(c)
        label = int(m['label']) if m else None
        vp = int(v >= args.vlm_threshold) if v is not None else None
        mp = int(m['pred']) if m else None
        if label is not None and vp is not None:
            labels.append(label); vlm_pred.append(vp); model_pred.append(mp); compared.append(c)
            right = {(True, True): 'both', (True, False): 'VLM', (False, True): 'model', (False, False): 'neither'}[
                (vp == label, mp == label)]
        elif m is None:
            right = 'ERROR: in VLM log but missing from model csv'
        else:
            right = 'model only: not analyzed by VLM'
        print(f"{c:>7s}  {'' if label is None else label:>5}  {'' if v is None else f'{v:.3f}':>9s} "
              f"{'' if vp is None else vp:>3}  {'' if m is None else m['votes_changed'] + '/' + m['votes_counted']:>11s} "
              f"{'' if mp is None else mp:>5}  {right}")

    errors = [c for c in cids if c in vlm and c not in model]
    model_only = [c for c in cids if c in model and c not in vlm]
    if errors:
        print(f"\nERROR: {len(errors)} clusters are in the VLM log but missing from the model csv: {errors}")
    if model_only:
        print(f"\nmodel only: {len(model_only)} clusters analyzed by the model but not the VLM "
              f"(left out of the side-by-side scores): {model_only}")
    print(f"\n{len(compared)} clusters compared ({sum(labels)} changed)")
    print(f"  {'':6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Spec':>6s} {'HM':>6s}")
    for name, preds in (('VLM', vlm_pred), ('model', model_pred)):
        print(f"  {name:6s} " + " ".join(f"{x:6.3f}" for x in scores(preds, labels)))
    vlm_only = [c for c, l, v, m in zip(compared, labels, vlm_pred, model_pred) if v == l and m != l]
    model_right = [c for c, l, v, m in zip(compared, labels, vlm_pred, model_pred) if m == l and v != l]
    print(f"\nVLM right, model wrong:  {len(vlm_only)}  clusters {vlm_only}")
    print(f"model right, VLM wrong:  {len(model_right)}  clusters {model_right}")