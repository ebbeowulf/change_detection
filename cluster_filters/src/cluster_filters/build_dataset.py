"""
Convert labeled change clusters into a parallel dataset of per-frame boxes.

For every run directory under <data_root>/<environment>/ that contains save_results* folders:
  1. Pick the "after" image directory whose orientation matches the renders (color/ or rotated/).
  2. Find the labels file for every prompt: <run>/<results_dir>/<variant>/<prompt>.labels.tight.json,
     or <prompt>.labels.json if there is no tight file (both label the same clusters).
  3. For each labeled cluster, load <prompt>_<cluster_id>.pkl and make one sample per frame listed in
     ds['all_images'] (the frames the pipeline found viewing the cluster). The box is the extent of the
     cluster's projected points plus a buffer.

Output, one directory per run:
  <output_root>/<environment>/<run path with / -> __>/
      meta.json                                    where the run's images, renders, depth and deltas are
      <results_dir>__<variant>__<prompt>.json      samples for one prompt
"""
import argparse
import fnmatch
import glob
import json
import os
import pickle
import numpy as np
from PIL import Image
from build_delta_change_masks import index_by_id, find_render, select_color_dir


# ---------------------------------------------------------------- small helpers

def jsonable(value):
    """Make a value safe for json: numpy scalars -> python numbers, anything else unknown -> string."""
    if hasattr(value, 'item'):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def key_to_id(key, color_ids, int_ids):
    """Map an all_images key (e.g. 21) to a color image ID (e.g. '00021').
    Accepts a bare ID or a filename ending in _<ID>; integer matching handles zero-padding."""
    stem = os.path.splitext(os.path.basename(str(key)))[0]
    vid = stem.rsplit('_', 1)[-1]
    if vid in color_ids:
        return vid
    if vid.isdigit() and int(vid) in int_ids:
        return int_ids[int(vid)]
    return None


# ---------------------------------------------------------------- finding runs and label files

def list_environments(args):
    """Environments named on the command line, or every directory under data_root."""
    if args.environments:
        return args.environments
    return sorted(d for d in os.listdir(args.data_root) if os.path.isdir(args.data_root + "/" + d))


def find_runs(env_dir, results_glob):
    """Any directory under env_dir holding at least one results dir is a change run.
    Follows symlinked directories, guarding against loops."""
    runs = []
    visited = set()
    for dirpath, dirnames, _ in os.walk(env_dir, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []
            continue
        visited.add(real)
        if any(fnmatch.fnmatch(d, results_glob) for d in dirnames):
            runs.append(dirpath)
            dirnames[:] = []  # don't descend into the run itself
    return sorted(runs)


def find_label_files(run_dir, args):
    """(results_dir, variant, variant_dir, prompt, labels_path) for every prompt in the run.
    Uses <prompt>.labels.tight.json when it exists, otherwise <prompt>.labels.json."""
    label_files = []
    for results_dir in sorted(glob.glob(run_dir + "/" + args.results_glob)):
        if not os.path.isdir(results_dir):
            continue
        for variant in sorted(os.listdir(results_dir)):
            vdir = results_dir + "/" + variant
            if not os.path.isdir(vdir):
                continue
            prompts = sorted({os.path.basename(p).split(".labels")[0] for p in glob.glob(vdir + "/*.labels*.json")})
            for prompt in prompts:
                tight = vdir + "/" + prompt + ".labels.tight.json"
                labels_path = tight if os.path.exists(tight) else vdir + "/" + prompt + ".labels.json"
                if os.path.exists(labels_path):
                    label_files.append((results_dir, variant, vdir, prompt, labels_path))
    return label_files


# ---------------------------------------------------------------- per-run image information

def prepare_run(run_dir, args):
    """Images available for this run, or None (with a warning) if the run can't be used.
    Returns a dict with: after_dir, depth_dir, color_ids {id: filename}, int_ids {int: id},
    valid_ids (ids that also have a render), img_size (w, h)."""
    if not os.path.isdir(run_dir + "/" + args.color_dir):
        print(f"  WARNING: no '{args.color_dir}' directory in {run_dir} - skipping run (set --color_dir)")
        return None

    # robot runs keep rotated copies; use whichever directory matches the renders' orientation
    after_dir, color_ids = select_color_dir(run_dir, args.color_dir, args.rotated_dir, args.renders_dir, args.render_prefix)
    if after_dir is None:
        print(f"  WARNING: neither {args.color_dir} nor {args.rotated_dir} matches the renders' orientation - skipping run")
        return None
    print(f"  using {after_dir}/")

    valid_ids = {v for v in color_ids if find_render(run_dir + "/" + args.renders_dir, args.render_prefix, v) is not None}
    if not valid_ids:
        print("  WARNING: no color images with matching renders - skipping run")
        return None

    return {'after_dir': after_dir,
            'depth_dir': args.depth_rotated_dir if after_dir == args.rotated_dir else args.depth_dir,
            'color_ids': color_ids,
            'int_ids': {int(v): v for v in color_ids if v.isdigit()},
            'valid_ids': valid_ids,
            'img_size': Image.open(run_dir + "/" + after_dir + "/" + color_ids[sorted(valid_ids)[0]]).size}


def write_meta(out_dir, run_dir, env, env_dir, run, args):
    """meta.json: where the loader finds this run's images, renders, depth and delta masks."""
    meta = {'source': os.path.abspath(run_dir), 'environment': env, 'run_path': os.path.relpath(run_dir, env_dir),
            'color_dir': run['after_dir'], 'renders_dir': args.renders_dir, 'render_prefix': args.render_prefix,
            'depth_dir': run['depth_dir'], 'delta_dir': args.delta_dir, 'image_size': list(run['img_size'])}
    with open(out_dir + "/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------- boxes for one cluster

def frame_box(cluster, params, pose, sx, sy, img_size, args):
    """Box [x0, y0, x1, y1] for one frame: extent of the cluster points projected by find_pts_in_image,
    rescaled to the image size, plus the buffer. None if fewer than min_points land in the image."""
    pts = np.array(cluster.find_pts_in_image(params, pose, num_points=args.num_points), dtype=float)
    if pts.shape[0] < args.min_points:
        return None, pts.shape[0]
    img_w, img_h = img_size
    (r0, c0), (r1, c1) = pts.min(0), pts.max(0)
    x0, x1, y0, y1 = c0 * sx, c1 * sx, r0 * sy, r1 * sy
    pad_x = max(args.box_buffer * (x1 - x0), args.buffer_px)
    pad_y = max(args.box_buffer * (y1 - y0), args.buffer_px)
    bbox = [int(max(0, np.floor(x0 - pad_x))), int(max(0, np.floor(y0 - pad_y))),
            int(min(img_w - 1, np.ceil(x1 + pad_x))), int(min(img_h - 1, np.ceil(y1 + pad_y)))]
    return bbox, pts.shape[0]


def cluster_samples(ds, run, args):
    """One sample per non-empty frame in ds['all_images'] that has an image, a render and a box."""
    params = ds['exp_params']['params']
    fl = ds['exp_params']['fList_new']
    cluster = ds['cluster']

    # projected points are in cam_params image space; scale them to the actual image size
    img_w, img_h = run['img_size']
    sx, sy = img_w / params.width, img_h / params.height
    if abs(sx - sy) > 0.01:
        raise ValueError(f"cam_params {params.width}x{params.height} and image {img_w}x{img_h} differ in aspect ratio")

    samples = []
    for key, entry in ds['all_images'].items():
        if not entry:
            continue                                     # empty entry: pipeline did not keep this frame
        vid = key_to_id(key, run['color_ids'], run['int_ids'])
        if vid is None:
            print(f"    WARNING: all_images key {key} has no matching color image")
            continue
        if vid not in run['valid_ids']:
            continue                                     # no render for this frame
        bbox, n_points = frame_box(cluster, params, fl.get_pose(key), sx, sy, run['img_size'], args)
        if bbox is None:
            continue
        samples.append({'frame_id': vid, 'pose_key': str(key), 'bbox': bbox, 'num_points': int(n_points),
                        'pt_count': jsonable(entry.get('pt_count')), 'max_prob': jsonable(entry.get('max_prob'))})
    return samples


# ---------------------------------------------------------------- one labels file -> one sample file

def convert_labels_file(labels_path, prompt, results_dir, variant, vdir, out_dir, run, args):
    """Write the sample file for one prompt (skipped if it exists, unless --overwrite 1)."""
    out_file = out_dir + "/" + os.path.basename(results_dir) + "__" + variant + "__" + prompt + ".json"
    if not args.overwrite and os.path.exists(out_file):
        return

    with open(labels_path) as f:
        labels = json.load(f)

    samples, header = [], None
    for cid, lab in labels.items():
        pkl_path = vdir + "/" + prompt + "_" + cid + ".pkl"
        if not os.path.exists(pkl_path):
            print(f"    WARNING: missing {pkl_path}")
            continue
        with open(pkl_path, 'rb') as f:
            ds = pickle.load(f)
        np.random.seed(args.seed)                        # find_pts_in_image samples randomly
        for s in cluster_samples(ds, run, args):
            s.update({'cluster_id': cid, 'changed': bool(lab.get('changed', False)), 'label': lab, 'label_source': 'human'})
            samples.append(s)
        if header is None:                               # run settings recorded once per file
            ep = ds['exp_params']
            header = {'query': jsonable(ds['query']), 'classifier': jsonable(ep['classifier']),
                      'detection_threshold': jsonable(ep['detection_threshold']), 'scale': jsonable(ep['scale'])}

    record = {'results_dir': os.path.basename(results_dir), 'run_variant': variant, 'prompt': prompt,
              'labels_file': os.path.abspath(labels_path),
              'num_points': args.num_points, 'box_buffer': args.box_buffer, 'buffer_px': args.buffer_px, 'seed': args.seed}
    record.update(header or {})
    record['samples'] = samples
    with open(out_file, 'w') as f:
        json.dump(record, f, indent=1)

    n_pos = sum(s['changed'] for s in samples)
    print(f"  {os.path.basename(results_dir)}/{variant}/{os.path.basename(labels_path)}: "
          f"{len(labels)} clusters -> {len(samples)} samples ({n_pos} changed)")


# ---------------------------------------------------------------- one run

def convert_run(run_dir, env, env_dir, args):
    run_name = os.path.relpath(run_dir, env_dir).replace(os.sep, "__")
    print(f"{env}/{run_name}")

    run = prepare_run(run_dir, args)
    if run is None:
        return
    label_files = find_label_files(run_dir, args)
    if not label_files:
        print(f"  WARNING: no labels files under {args.results_glob}/*/ - skipping run")
        return

    out_dir = args.output_root + "/" + env + "/" + run_name
    os.makedirs(out_dir, exist_ok=True)
    write_meta(out_dir, run_dir, env, env_dir, run, args)
    for results_dir, variant, vdir, prompt, labels_path in label_files:
        convert_labels_file(labels_path, prompt, results_dir, variant, vdir, out_dir, run, args)


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description='Convert labeled change clusters into a parallel <environment>/<run> dataset of per-frame boxes')
    parser.add_argument('data_root', type=str, help='root holding one directory per environment (e.g. /data2/datasets)')
    parser.add_argument('output_root', type=str, help='root of the parallel dataset (e.g. /data2/datasets/change_dataset)')
    parser.add_argument('--environments', type=str, nargs='*', default=[], help='environments to convert (default = every directory under data_root)')
    parser.add_argument('--results_glob', type=str, default='save_results*', help='pattern matching results directories inside a run (default=save_results*)')
    parser.add_argument('--color_dir', type=str, default='color', help='name of the captured "after" image subdirectory (default=color)')
    parser.add_argument('--rotated_dir', type=str, default='rotated', help='name of the rotated image subdirectory used when color images do not match the renders in orientation (default=rotated)')
    parser.add_argument('--depth_rotated_dir', type=str, default='depth_rotated', help='name of the rotated depth subdirectory, recorded in meta.json when rotated images are used (default=depth_rotated)')
    parser.add_argument('--renders_dir', type=str, default='renders', help='name of the Gaussian Splatting "before" render subdirectory (default=renders)')
    parser.add_argument('--render_prefix', type=str, default='rgb', help='filename prefix of the color renders (default=rgb)')
    parser.add_argument('--depth_dir', type=str, default='depth', help='name of the depth image subdirectory, recorded in meta.json (default=depth)')
    parser.add_argument('--delta_dir', type=str, default='delta_change', help='name of the delta mask subdirectory, recorded in meta.json (default=delta_change)')
    parser.add_argument('--num_points', type=int, default=20, help='points sampled per frame by find_pts_in_image (default=20)')
    parser.add_argument('--min_points', type=int, default=3, help='minimum projected points for a frame to produce a sample (default=3)')
    parser.add_argument('--box_buffer', type=float, default=0.25, help='fraction of box width/height added on each side (default=0.25)')
    parser.add_argument('--buffer_px', type=int, default=10, help='minimum buffer in pixels on each side (default=10)')
    parser.add_argument('--seed', type=int, default=0, help='random seed set before each cluster so boxes are reproducible (default=0)')
    parser.add_argument('--overwrite', type=int, default=0, help='1 = rebuild sample files that already exist (default=0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for env in list_environments(args):
        env_dir = args.data_root + "/" + env
        if not os.path.isdir(env_dir):
            print(f"ERROR: {env_dir} does not exist. data_root should be the directory that CONTAINS "
                  f"the environments (e.g. /data2/datasets), not an environment itself - skipping {env}")
            continue
        runs = find_runs(env_dir, args.results_glob)
        print(f"{env}: found {len(runs)} runs containing {args.results_glob}")
        for run_dir in runs:
            convert_run(run_dir, env, env_dir, args)