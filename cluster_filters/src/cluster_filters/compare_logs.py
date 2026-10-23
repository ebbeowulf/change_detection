"""
Side-by-side comparison of the VLM filter and the DNN for ONE run and ONE prompt.

  python compare_logs.py <vlm_log.json> <predictions.csv> <run> <prompt> --image_threshold S

Inputs
  vlm_log          the VLM log for this run/prompt; uses its "scores"
                   (per cluster: fraction of views the VLM called a new object)
  predictions.csv  raw per-image DNN scores written by train_change_filter.py (predictions.csv or val_predictions.csv)
  run              the run as in predictions.csv (e.g. j234/T1__changes__changes5), or any unique part of it
  prompt           e.g. electronics

Cluster scores (same form for both methods)
  VLM   fraction of views the VLM called a new object; changed if >= --vlm_threshold (default 0.5)
  DNN   fraction of the cluster's images with DNN score >= --image_threshold; changed if >= --cluster_threshold (default 0.5)
The DNN scores are raw, so rerunning with different --image_threshold values traces out its ROC.

Coverage
  ERROR      cluster is in the VLM log but has no DNN images
  DNN only   cluster has DNN images but no VLM score
  Both are listed but left out of the side-by-side counts.

Summary csv (--output_csv)
  One row per run/prompt: run, prompt, the three thresholds, and TP/TN/FP/FN for the VLM and the DNN.
  Rerunning the same run/prompt/thresholds replaces its row.
"""
import argparse
import csv
import json

SUMMARY_KEY = ['run', 'prompt', 'image_threshold', 'cluster_threshold', 'vlm_threshold']


def confusion(preds, labels):
    """(tp, tn, fp, fn)"""
    tp = sum(p and l for p, l in zip(preds, labels))
    tn = sum(not p and not l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum(l and not p for p, l in zip(preds, labels))
    return tp, tn, fp, fn


def write_summary(path, row):
    """Add row to the summary csv, replacing an earlier row for the same run/prompt/thresholds."""
    rows = []
    try:
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if [r[k] for k in SUMMARY_KEY] != [str(row[k]) for k in SUMMARY_KEY]]
    except FileNotFoundError:
        pass
    rows.append(row)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)


def load_dnn(predictions_csv, run, prompt, image_threshold):
    """Per cluster: label, number of images, number with score >= image_threshold. Also the matched run name."""
    with open(predictions_csv) as f:
        all_rows = list(csv.DictReader(f))
    run_rows = [r for r in all_rows if run in r['run']]
    rows = [r for r in run_rows if r['prompt'] == prompt]
    if not rows:
        raise SystemExit(f"no DNN predictions for {run} / {prompt}"
                         f" - this run has predictions for: {sorted({r['prompt'] for r in run_rows})}")
    runs, sources = sorted({r['run'] for r in rows}), sorted({r['source_file'] for r in rows})
    if len(runs) > 1 or len(sources) > 1:
        raise SystemExit(f"'{run}' / '{prompt}' matches more than one run or label source: {runs} {sources}")
    dnn = {}
    for r in rows:
        c = dnn.setdefault(r['cluster_id'], {'label': int(r['label']), 'n': 0, 'yes': 0})
        c['n'] += 1
        c['yes'] += int(float(r['score']) >= image_threshold)
    return dnn, runs[0], sources[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare the VLM log and the DNN cluster by cluster for one run and prompt')
    parser.add_argument('vlm_log', type=str, help='VLM log (json) for this run/prompt')
    parser.add_argument('predictions_csv', type=str, help='predictions.csv written by train_change_filter.py')
    parser.add_argument('run', type=str, help='run as in predictions.csv, or a unique part of it')
    parser.add_argument('prompt', type=str, help='prompt, e.g. electronics')
    parser.add_argument('--image_threshold', type=float, required=True, help='DNN confidence: an image counts as changed if its score >= this')
    parser.add_argument('--cluster_threshold', type=float, default=0.5, help='DNN: a cluster is changed if its fraction of changed images >= this (default=0.5)')
    parser.add_argument('--vlm_threshold', type=float, default=0.5, help='VLM: a cluster is changed if its fraction of new-object views >= this (default=0.5)')
    parser.add_argument('--output_csv', type=str, default='', help='add a summary row for this comparison to this csv (default = print only)')
    args = parser.parse_args()

    with open(args.vlm_log) as f:
        vlm = {str(k): float(v) for k, v in json.load(f)['scores'].items()}
    dnn, run, source = load_dnn(args.predictions_csv, args.run, args.prompt, args.image_threshold)
    print(f"run {run}   label source {source}")
    print(f"DNN: image changed if score >= {args.image_threshold}, cluster changed if fraction >= {args.cluster_threshold}   "
          f"VLM: cluster changed if score >= {args.vlm_threshold}\n")

    # one line per cluster
    cids = sorted(set(vlm) | set(dnn), key=lambda c: int(c) if c.isdigit() else c)
    print(f"{'cluster':>7s} {'label':>5s} | {'VLM':>5s} {'dec':>3s} | {'images':>6s} {'changed':>7s} {'DNN':>5s} {'dec':>3s} | outcome")
    labels, vlm_dec, dnn_dec, compared, errors, dnn_only = [], [], [], [], [], []
    for c in cids:
        v, d = vlm.get(c), dnn.get(c)
        d_score = d['yes'] / d['n'] if d else None
        vd = int(v >= args.vlm_threshold) if v is not None else None
        dd = int(d_score >= args.cluster_threshold) if d else None
        if d is None:
            outcome = 'ERROR: in VLM log, no DNN images'
            errors.append(c)
        elif v is None:
            outcome = 'DNN only: no VLM score'
            dnn_only.append(c)
        else:
            labels.append(d['label']); vlm_dec.append(vd); dnn_dec.append(dd); compared.append(c)
            outcome = {(True, True): 'both right', (True, False): 'VLM right', (False, True): 'DNN right',
                       (False, False): 'both wrong'}[(vd == d['label'], dd == d['label'])]
        print(f"{c:>7s} {'' if d is None else d['label']:>5} | {'' if v is None else f'{v:.2f}':>5s} "
              f"{'' if vd is None else vd:>3} | {'' if d is None else d['n']:>6} {'' if d is None else d['yes']:>7} "
              f"{'' if d is None else f'{d_score:.2f}':>5s} {'' if dd is None else dd:>3} | {outcome}")
    if errors:
        print(f"\nERROR: {len(errors)} clusters in the VLM log have no DNN images: {errors}")
    if dnn_only:
        print(f"DNN only: {len(dnn_only)} clusters have DNN images but no VLM score: {dnn_only}")

    # counts
    summary = {'run': run, 'prompt': args.prompt, 'image_threshold': args.image_threshold,
               'cluster_threshold': args.cluster_threshold, 'vlm_threshold': args.vlm_threshold}
    print(f"\n{len(compared)} clusters compared ({sum(labels)} changed)")
    print(f"  {'':4s} {'TP':>4s} {'TN':>4s} {'FP':>4s} {'FN':>4s}")
    for name, dec in (('vlm', vlm_dec), ('dnn', dnn_dec)):
        tp, tn, fp, fn = confusion(dec, labels)
        print(f"  {name.upper():4s} {tp:4d} {tn:4d} {fp:4d} {fn:4d}")
        summary.update({f"{name}_tp": tp, f"{name}_tn": tn, f"{name}_fp": fp, f"{name}_fn": fn})
    print(f"\nVLM right, DNN wrong: {[c for c, l, v, d in zip(compared, labels, vlm_dec, dnn_dec) if v == l and d != l]}")
    print(f"DNN right, VLM wrong: {[c for c, l, v, d in zip(compared, labels, vlm_dec, dnn_dec) if d == l and v != l]}")

    if args.output_csv:
        write_summary(args.output_csv, summary)
        print(f"\nsummary row written to {args.output_csv}")