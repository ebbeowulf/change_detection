import argparse
import csv
import glob
import json
import os
import numpy as np
import cv2
from PIL import Image
from dino_features import add_image_paths

GREEN, RED, BLUE = (0, 200, 0), (220, 0, 0), (0, 150, 255)


def find_run(dataset_root, target):
    """Accept a dataset run name (e.g. j234/T3__changes__changes5) or an original run directory."""
    if os.path.isfile(dataset_root + "/" + target + "/meta.json"):
        return target.strip("/")
    want = os.path.realpath(target)
    for meta_path in glob.glob(dataset_root + "/*/*/meta.json"):
        with open(meta_path) as f:
            if os.path.realpath(json.load(f)['source']) == want:
                return os.path.relpath(os.path.dirname(meta_path), dataset_root)
    raise ValueError(f"no run in {dataset_root} has source {target}")


def load_predictions(results_dir):
    probs, clusters = {}, {}
    with open(results_dir + "/test_predictions.csv") as f:
        for x in csv.DictReader(f):
            probs[(x['run'], x['source_file'], x['cluster_id'], x['frame_id'])] = (float(x['prob']), float(x['frac_above']))
    with open(results_dir + "/test_clusters.csv") as f:
        for x in csv.DictReader(f):
            clusters[(x['run'], x['source_file'], x['cluster_id'])] = x
    return probs, clusters


def build_clusters(dataset_root, run, prompt, source, probs, cluster_preds, show_unscored):
    """One entry per cluster: its label, cluster-level prediction, and the frames to display."""
    out = []
    for path in sorted(glob.glob(dataset_root + "/" + run + "/*.json")):
        fname = os.path.basename(path)
        if fname == 'meta.json' or not (fname.endswith("__" + prompt + ".json") or ("__" + prompt + "__") in fname):
            continue
        if source and source not in fname:
            continue
        with open(path) as f:
            rec = json.load(f)
        by_cluster = {}
        for s in rec['samples']:
            label_name = s['label'].get('name', '?') if isinstance(s.get('label'), dict) else '?'
            s = dict(s, run=run, source_file=fname, label=int(s['changed']), label_name=label_name)
            p = probs.get((run, fname, s['cluster_id'], s['frame_id']))
            s['prob'], s['frac'] = p if p else (None, None)
            if p is None and not show_unscored:
                continue
            by_cluster.setdefault(s['cluster_id'], []).append(s)
        for cid in sorted(by_cluster, key=lambda c: int(c) if c.isdigit() else c):
            pred = cluster_preds.get((run, fname, cid))
            if pred is None and not show_unscored:
                continue
            frames = sorted(by_cluster[cid], key=lambda s: s['frame_id'])
            out.append({'source_file': fname, 'cluster_id': cid, 'frames': frames,
                        'name': frames[0]['label_name'], 'pred': pred})
    return out


def draw_box(img, bbox, color, width):
    out = img.copy()
    cv2.rectangle(out, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, width)
    return out


def crop_view(img, bbox, context):
    """With context > 0, return a square window context x the longer box side around the box,
    and the box shifted into that window; otherwise the full image."""
    if context <= 0:
        return img, bbox
    x0, y0, x1, y1 = bbox
    half = context * max(x1 - x0, y1 - y0) / 2.0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    wx0, wy0 = max(int(cx - half), 0), max(int(cy - half), 0)
    wx1, wy1 = min(int(cx + half), img.shape[1]), min(int(cy + half), img.shape[0])
    return img[wy0:wy1, wx0:wx1], [x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0]


def render_row(axes, frame, threshold, meta, context, col_titles):
    """One matrix row: before render | after image | delta, each with the box colored by the image's prediction."""
    after = np.asarray(Image.open(frame['after_path']).convert('RGB'))
    before_img = Image.open(frame['before_path']).convert('RGB')
    if before_img.size != (after.shape[1], after.shape[0]):
        before_img = before_img.resize((after.shape[1], after.shape[0]), Image.BILINEAR)
    before = np.asarray(before_img)
    if frame['prob'] is None:
        color, verdict = BLUE, "not scored (filtered)"
    elif frame['prob'] >= threshold:
        color, verdict = GREEN, f"CHANGED p={frame['prob']:.3f}"
    else:
        color, verdict = RED, f"not changed p={frame['prob']:.3f}"
    frac = "" if frame['frac'] is None else f"  frac={frame['frac']:.2f}"

    delta_path = (meta['source'] + "/" + meta['delta_dir'] + "/" + meta['_delta_model'] + "/"
                  + meta['_prompt'] + "_" + frame['frame_id'] + ".npy")
    for ax in axes:
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    b_view, box = crop_view(before, frame['bbox'], context)
    a_view, _ = crop_view(after, frame['bbox'], context)
    width = max(2, a_view.shape[1] // 150)
    axes[0].imshow(draw_box(b_view, box, color, width))
    axes[1].imshow(draw_box(a_view, box, color, width))
    if os.path.exists(delta_path):
        d_view, _ = crop_view(np.load(delta_path).astype(np.float32), frame['bbox'], context)
        axes[2].imshow(d_view, cmap='gray', vmin=-1, vmax=1)
        x0, y0, x1, y1 = box
        axes[2].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=np.array(color) / 255.0, linewidth=2)
    else:
        axes[2].text(0.5, 0.5, "no delta file", ha='center', va='center', transform=axes[2].transAxes)
    # row label on the left, colored like the box
    axes[0].set_ylabel(f"frame {frame['frame_id']}\n{verdict}{frac}", color=np.array(color) / 255.0,
                       fontsize=9, rotation=0, ha='right', va='center')
    if col_titles:
        axes[0].set_title("before (render)")
        axes[1].set_title("after")
        axes[2].set_title("delta, bright = increase")


def cluster_header(cluster):
    pred = cluster['pred']
    truth = "CHANGED" if cluster['frames'][0]['label'] else "not changed"
    if pred is None:
        cl = "cluster not evaluated"
    else:
        cl = (f"cluster prediction: {'CHANGED' if pred['pred'] == '1' else 'not changed'} "
              f"({pred['votes_changed']}/{pred['votes_counted']} votes)")
    return f"{cluster['source_file']}  cluster {cluster['cluster_id']} ({cluster['name']})  label: {truth}  |  {cl}"


def render_page(fig, axes, cluster, page, rows, threshold, meta, context, prefix=""):
    frames = cluster['frames'][page * rows:(page + 1) * rows]
    for r in range(rows):
        if r < len(frames):
            render_row(axes[r], frames[r], threshold, meta, context, col_titles=(r == 0))
        else:
            for ax in axes[r]:
                ax.clear()
                ax.axis('off')
    n_pages = (len(cluster['frames']) + rows - 1) // rows
    fig.suptitle(f"{prefix}page {page + 1}/{n_pages}, frames {page * rows + 1}-{page * rows + len(frames)} "
                 f"of {len(cluster['frames'])}\n" + cluster_header(cluster), fontsize=11)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step through the clusters of one prompt in one run as pages of images (rows: frames; columns: before, after, delta) with each box colored by its prediction')
    parser.add_argument('dataset_root', type=str, help='root of the change dataset (e.g. /data2/datasets/change_dataset)')
    parser.add_argument('target', type=str, help='run to view: dataset run name (j234/T3__changes__changes5) or original run directory')
    parser.add_argument('prompt', type=str, help='prompt to view (e.g. electronics or "small items")')
    parser.add_argument('results_dir', type=str, help='training output directory holding test_predictions.csv and test_clusters.csv')
    parser.add_argument('--source', type=str, default='', help='only label sources whose name contains this (e.g. save_results_unfiltered__sam3_0.1_openVocab)')
    parser.add_argument('--threshold', type=float, default=-1.0, help='probability threshold for an image to count as changed (default = value used in training, from metrics.json)')
    parser.add_argument('--delta_model', type=str, default='', help='delta_change/<model> to display (default = value used in training)')
    parser.add_argument('--show_unscored', type=int, default=1, help='1 = also show frames removed by the visibility filter, boxed in blue (default=1)')
    parser.add_argument('--rows', type=int, default=5, help='images (matrix rows) per page (default=5)')
    parser.add_argument('--context', type=float, default=0.0, help='0 = show full images; >0 = zoom to a window this many times the box size around the box (e.g. 3) (default=0)')
    parser.add_argument('--save_dir', type=str, default='', help='save every page as PNG here instead of opening an interactive window')
    args = parser.parse_args()

    with open(args.results_dir + "/metrics.json") as f:
        train_args = json.load(f)['args']
    threshold = args.threshold if args.threshold >= 0 else train_args.get('threshold', 0.5)
    delta_model = args.delta_model if args.delta_model else train_args.get('delta_model', 'sam3')
    prompt = args.prompt.strip().lower().replace(' ', '_')

    run = find_run(args.dataset_root, args.target)
    with open(args.dataset_root + "/" + run + "/meta.json") as f:
        meta = json.load(f)
    meta['_delta_model'], meta['_prompt'] = delta_model, prompt
    probs, cluster_preds = load_predictions(args.results_dir)
    clusters = build_clusters(args.dataset_root, run, prompt, args.source, probs, cluster_preds, args.show_unscored)
    for c in clusters:
        c['frames'] = add_image_paths(args.dataset_root, c['frames'])
    clusters = [c for c in clusters if c['frames']]
    if not clusters:
        raise SystemExit(f"no clusters for prompt '{prompt}' in {run}" + (f" matching {args.source}" if args.source else ""))
    print(f"{run}: {len(clusters)} clusters for '{prompt}' (threshold {threshold})")
    for c in clusters:
        print(f"  {c['source_file']} cluster {c['cluster_id']}: {len(c['frames'])} frames")

    import matplotlib
    if args.save_dir:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = args.rows
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3.0 * rows + 1.2), squeeze=False)
    fig.subplots_adjust(left=0.16, right=0.99, top=1 - 1.1 / (3.0 * rows + 1.2), bottom=0.01, wspace=0.03, hspace=0.08)

    if args.save_dir:
        for c in clusters:
            out_dir = args.save_dir + "/" + c['source_file'][:-5]
            os.makedirs(out_dir, exist_ok=True)
            for page in range((len(c['frames']) + rows - 1) // rows):
                render_page(fig, axes, c, page, rows, threshold, meta, args.context)
                fig.savefig(out_dir + "/cluster_" + c['cluster_id'] + "_page" + str(page + 1) + ".png", dpi=80)
        print(f"saved to {args.save_dir}")
    else:
        state = {'c': 0, 'p': 0}

        def show():
            c = clusters[state['c']]
            render_page(fig, axes, c, state['p'], rows, threshold, meta, args.context,
                        prefix=f"[cluster {state['c'] + 1}/{len(clusters)}]  <-/-> page, up/down cluster, q quit   ")
            fig.canvas.draw_idle()

        def on_key(event):
            n_pages = (len(clusters[state['c']]['frames']) + rows - 1) // rows
            if event.key == 'right':
                state['p'] = (state['p'] + 1) % n_pages
            elif event.key == 'left':
                state['p'] = (state['p'] - 1) % n_pages
            elif event.key in ('up', 'down'):
                state['c'] = (state['c'] + (1 if event.key == 'up' else -1)) % len(clusters)
                state['p'] = 0
            elif event.key == 'q':
                plt.close(fig)
                return
            show()

        fig.canvas.mpl_connect('key_press_event', on_key)
        show()
        plt.show()