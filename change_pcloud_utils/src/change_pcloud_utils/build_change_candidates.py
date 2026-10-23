"""
Step 3 of the change pipeline (robot runs only): find candidate objects as 3D clusters and record every
image that views each one.

For one run directory:
  1. For every AFTER image and every prompt, run the segmentation model on the AFTER image and on its
     render (the BEFORE view), and save the delta map  delta = AFTER - BEFORE.
  2. Collect candidate points in 3D from the depth image:
        --candidates openVocab   pixels where the AFTER score  > --threshold   (weighted by that score)
        --candidates change      pixels where the delta        > --threshold   (weighted by the delta)
     With SAM3 the score map is the detection confidence (>= --sam3_confidence) or 0, so in openVocab mode
     any --threshold below --sam3_confidence simply means "every detected pixel".
  3. Cluster each prompt's points: merge into --voxel_size voxels (weights summed), weighted DBSCAN
     (--dbscan_eps, --dbscan_min_samples), keep clusters of at least --min_cluster_voxels voxels.
  4. For each cluster, its views are the images whose camera sees the cluster centroid; the box in each
     view is the extent of the cluster's projected points plus a buffer (same as the filter's training data).
  5. Write one record per cluster. No thresholds are applied to views; filters decide later.

Fixed robot inputs (inside <root_dir>):
  images_combined/   AFTER color images (COLMAP-registered, frame keyword 'new')
  depth_rotated/     AFTER depth images
  renders/           BEFORE color renders (rgb_<ID>.png)
  colmap_combined/sparse_combined/0/   cameras.txt, images.txt

Output (everything this program writes stays inside <root_dir>/<save_dir>; nothing else is touched):
  <root_dir>/<save_dir>/                                           (save_dir defaults to change2.0_results)
      delta_<segmenter>/<prompt>_<ID>.npy   delta maps (float16), read by the filters. <segmenter> is
                                            sam3_<sam3_confidence> or clipseg; the same for every
                                            --threshold / --candidates setting
      <classifier>_<threshold>_<candidates>/
      run_settings.json              settings, inputs, skipped images, format version
      <prompt>.candidates.npz        candidate points and weights per prompt (cache, so clustering can be redone alone)
      clusters/<prompt>_<NNNN>.json  one record per cluster: uid, prompt, 3D box, centroid, stats, views
      clusters/<prompt>_<NNNN>.npz   the cluster's 3D points
  Cluster records are plain JSON/NPZ so any environment can read them without change_pcloud_utils.
"""
import argparse
import glob
import json
import os
import time
import numpy as np
import cv2
import torch
from PIL import Image
from change_pcloud_utils.colmap_utils import get_camera_params, build_file_list, build_rendered_file_list
from change_pcloud_utils.pcloud_creation_utils import pcloud_change
from change_pcloud_utils.map_utils import identify_related_images_global_pose, get_weighted_clusters, DEVICE

FORMAT_VERSION = 1
COLOR_DIR = 'images_combined'
DEPTH_DIR = 'depth_rotated'
RENDERS_DIR = 'renders'
COLMAP_DIR = 'colmap_combined/sparse_combined/0'
FRAME_KEYWORD = 'new'


# ---------------------------------------------------------------- small helpers

def prompt_slug(prompt):
    """'Small items' -> 'small_items' (used in every file name)."""
    return prompt.strip().lower().replace(' ', '_')


def frame_id(fList, key):
    """Image ID from the color file name: .../new_00042.png -> '00042'."""
    return os.path.splitext(os.path.basename(fList.get_color_fileName(key)))[0].rsplit('_', 1)[-1]


def jsonable(value):
    """numpy scalars/arrays -> plain python, so records can be written as JSON."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return value


def to_numpy(t):
    if t is None:
        return None
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


# ---------------------------------------------------------------- setup and input checks

def load_run(args):
    """Camera parameters and the AFTER / render file lists, after checking the robot inputs exist."""
    root = args.root_dir
    for d in (COLOR_DIR, DEPTH_DIR, RENDERS_DIR, COLMAP_DIR):
        if not os.path.isdir(root + "/" + d):
            raise SystemExit(f"ERROR: missing {root}/{d}")
    params = get_camera_params(root + "/" + COLMAP_DIR + "/", args.nerfacto_dir)
    fList_new = build_file_list(root + "/" + COLOR_DIR + "/", root + "/" + DEPTH_DIR + "/", args.out_dir,
                                root + "/" + COLMAP_DIR + "/", FRAME_KEYWORD)
    fList_renders = build_rendered_file_list(fList_new, root + "/" + RENDERS_DIR + "/", args.out_dir)
    if len(fList_new.keys()) == 0:
        raise SystemExit(f"ERROR: no '{FRAME_KEYWORD}' images registered in {root}/{COLMAP_DIR}/images.txt")
    return params, fList_new, fList_renders


def check_image_sizes(fList_new, fList_renders):
    """AFTER image, depth and render must share one size and orientation, or every box would be wrong."""
    key = sorted(fList_new.keys())[0]
    sizes = {'color': Image.open(fList_new.get_color_fileName(key)).size,
             'depth': Image.open(fList_new.get_depth_fileName(key)).size,
             'render': Image.open(fList_renders.get_color_fileName(key)).size}
    if len(set(sizes.values())) != 1:
        raise SystemExit(f"ERROR: image sizes differ for frame {key}: {sizes} - check orientation of the inputs")
    return sizes['color']


# ---------------------------------------------------------------- 1-2: segmentation, deltas, candidate points

def build_candidate_clouds(params, fList_new, fList_renders, prompts, args, skipped):
    """Run segmentation on every AFTER image and its render, save the delta maps, and gather candidate
    points and weights per prompt. Returns {prompt: (xyz (N,3) float32, weights (N,) float32)}."""
    xyz = {p: [] for p in prompts}
    weights = {p: [] for p in prompts}
    delta_dir = args.delta_dir
    os.makedirs(delta_dir, exist_ok=True)

    creator = pcloud_change(params)
    creator.setup_segmentation(prompts, args.classifier, args.sam3_confidence)
    for key in sorted(fList_new.keys()):
        try:
            after = Image.open(fList_new.get_color_fileName(key))
            before = Image.open(fList_renders.get_color_fileName(key))
            depth = cv2.imread(fList_new.get_depth_fileName(key), -1)
        except Exception as e:
            skipped.append({'frame': str(key), 'reason': f"could not load images: {e}"})
            continue
        if depth is None or not creator.load_image(after, depth, fList_new.get_pose(key), str(key)):
            skipped.append({'frame': str(key), 'reason': 'depth or image failed to load'})
            continue

        result = creator.segment_pair(prompts, before)
        fid = frame_id(fList_new, key)
        for prompt in prompts:
            new_p = result[prompt]['new_prob']
            base_p = result[prompt]['baseline_prob']
            if new_p is None:                                       # nothing detected in the AFTER image
                new_p = torch.zeros(creator.loaded_image['depthT'].shape, dtype=float, device=DEVICE)
            if base_p is None:
                base_p = torch.zeros_like(new_p)
            delta = new_p - base_p
            np.save(delta_dir + "/" + prompt_slug(prompt) + "_" + fid + ".npy", to_numpy(delta).astype(np.float16))

            weight = delta if args.candidates == 'change' else new_p
            mask = creator.loaded_image['depth_mask'] * (weight > args.threshold)
            if mask.sum() == 0:
                continue
            pts = creator.get_pts(weight, mask)
            xyz[prompt].append(to_numpy(pts['xyz']).astype(np.float32))
            weights[prompt].append(to_numpy(pts['probs']).astype(np.float32))
        print(f"  frame {fid} done")
    return {p: (np.concatenate(xyz[p]) if xyz[p] else np.zeros((0, 3), np.float32),
                np.concatenate(weights[p]) if weights[p] else np.zeros(0, np.float32)) for p in prompts}


def cluster_prompt(xyz, weights, args):
    """Weighted DBSCAN clusters for one prompt's candidate points (points with NaN coordinates dropped)."""
    keep = ~np.isnan(xyz).any(axis=1)
    pts = torch.tensor(xyz[keep], dtype=torch.float64, device=DEVICE)
    w = torch.tensor(weights[keep], dtype=torch.float64, device=DEVICE)
    return get_weighted_clusters(pts, w, args.voxel_size, args.dbscan_eps, args.dbscan_min_samples, args.min_cluster_voxels)


# ---------------------------------------------------------------- 4: views and boxes for one cluster

def cluster_views(cluster, params, fList_new, img_size, args):
    """Every image whose camera sees the cluster centroid, with the box the filters were trained on:
    extent of the cluster points projected by find_pts_in_image, rescaled to the image, plus the buffer."""
    img_w, img_h = img_size
    sx, sy = img_w / params.width, img_h / params.height
    views = []
    for key in identify_related_images_global_pose(params, fList_new, cluster.centroid):
        key = int(key)
        pts = np.array(cluster.find_pts_in_image(params, fList_new.get_pose(key), num_points=args.num_points), dtype=float)
        if pts.shape[0] < args.min_points:
            continue                                                 # too few points to draw a box
        (r0, c0), (r1, c1) = pts.min(0), pts.max(0)
        x0, x1, y0, y1 = c0 * sx, c1 * sx, r0 * sy, r1 * sy
        pad_x = max(args.box_buffer * (x1 - x0), args.buffer_px)
        pad_y = max(args.box_buffer * (y1 - y0), args.buffer_px)
        bbox = [int(max(0, np.floor(x0 - pad_x))), int(max(0, np.floor(y0 - pad_y))),
                int(min(img_w - 1, np.ceil(x1 + pad_x))), int(min(img_h - 1, np.ceil(y1 + pad_y)))]
        views.append({'frame_id': frame_id(fList_new, key), 'pose_key': key,
                      'image': os.path.basename(fList_new.get_color_fileName(key)),
                      'bbox': bbox, 'points_in_frame': int(pts.shape[0])})
    return views


# ---------------------------------------------------------------- 5: cluster records

def write_cluster(cluster_dir, prompt, index, cluster, views):
    """clusters/<prompt>_<NNNN>.json (record) + .npz (3D points)."""
    name = prompt_slug(prompt) + "_" + f"{index:04d}"
    record = {'format_version': FORMAT_VERSION,
              'uid': prompt_slug(prompt) + "/" + f"{index:04d}",
              'prompt': prompt,
              'box3d': jsonable(np.asarray(cluster.box)),
              'centroid': jsonable(np.asarray(cluster.centroid).ravel()),
              'n_points': int(np.asarray(cluster.pts).shape[0]),
              'prob_stats': jsonable(getattr(cluster, 'prob_stats', {}) or {}),
              'n_views': len(views),
              'views': views}
    with open(cluster_dir + "/" + name + ".json", 'w') as f:
        json.dump(record, f, indent=1)
    np.savez_compressed(cluster_dir + "/" + name + ".npz", points=np.asarray(cluster.pts, dtype=np.float32))


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description='Step 3: build 3D change candidates and their views for one robot run')
    parser.add_argument('nerfacto_dir', type=str, help='nerfstudio directory containing dataparser_transforms.json')
    parser.add_argument('root_dir', type=str, help='run directory holding images_combined, depth_rotated, renders and colmap_combined')
    parser.add_argument('--candidates', type=str, default='openVocab', help="cluster on 'openVocab' (AFTER scores) or 'change' (delta) points (default=openVocab)")
    parser.add_argument('--classifier', type=str, default='sam3', help="segmentation model: 'sam3' or 'clipseg' (default=sam3)")
    parser.add_argument('--sam3_confidence', type=float, default=0.1, help='SAM3 only: detections below this confidence are dropped (default=0.1)')
    parser.add_argument('--threshold', type=float, default=0.1, help='score (openVocab) or delta (change) a pixel must exceed to become a candidate point (default=0.1)')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='voxel size in meters for clustering (default=0.01)')
    parser.add_argument('--dbscan_eps', type=float, default=0.05, help='DBSCAN neighborhood radius in meters (default=0.05)')
    parser.add_argument('--dbscan_min_samples', type=int, default=2200, help='DBSCAN core requirement: summed voxel weight within eps (default=2200)')
    parser.add_argument('--min_cluster_voxels', type=int, default=100, help='clusters with fewer voxels are dropped (default=100, about 100 cm2 of surface at 1 cm voxels)')
    parser.add_argument('--queries', type=str, nargs='*', default=['clothing', 'dishes', 'electronics', 'general clutter', 'small items', 'trash'],
                        help='prompts to search for (default: clothing dishes electronics "general clutter" "small items" trash)')
    parser.add_argument('--save_dir', type=str, default='change2.0_results', help='subdirectory of root_dir that holds all outputs (default=change2.0_results)')
    parser.add_argument('--num_points', type=int, default=20, help='cluster points projected per view to form its box (default=20, as in training)')
    parser.add_argument('--min_points', type=int, default=3, help='projected points needed to draw a box (default=3)')
    parser.add_argument('--box_buffer', type=float, default=0.25, help='fraction of box width/height added on each side (default=0.25, as in training)')
    parser.add_argument('--buffer_px', type=int, default=10, help='minimum buffer in pixels on each side (default=10, as in training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed set before each cluster so boxes are reproducible (default=0)')
    parser.add_argument('--rebuild', type=int, default=0, help='1 = rerun segmentation even if cached candidate points exist (default=0)')
    args = parser.parse_args()
    if args.candidates not in ('openVocab', 'change'):
        parser.error("--candidates must be 'openVocab' or 'change'")
    if args.classifier not in ('sam3', 'clipseg'):
        parser.error("--classifier must be 'sam3' or 'clipseg'")
    args.root_dir = args.root_dir.rstrip('/')
    args.out_dir = args.root_dir + "/" + args.save_dir + "/" + f"{args.classifier}_{args.threshold}_{args.candidates}"
    # deltas depend only on the segmenter (and SAM3's detection confidence), not on --threshold or --candidates
    segmenter = f"sam3_{args.sam3_confidence}" if args.classifier == 'sam3' else args.classifier
    args.delta_dir = args.root_dir + "/" + args.save_dir + "/delta_" + segmenter
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = [q.strip().lower() for q in args.queries]

    params, fList_new, fList_renders = load_run(args)
    img_size = check_image_sizes(fList_new, fList_renders)
    print(f"{args.root_dir}: {len(fList_new.keys())} AFTER images {img_size[0]}x{img_size[1]}, "
          f"{args.classifier} {args.candidates} > {args.threshold}")

    # 1-2: candidate points (cached per prompt so clustering can be rerun without segmentation)
    skipped = []
    cloud_files = {p: args.out_dir + "/" + prompt_slug(p) + ".candidates.npz" for p in prompts}
    if args.rebuild or not all(os.path.exists(f) for f in cloud_files.values()):
        clouds = build_candidate_clouds(params, fList_new, fList_renders, prompts, args, skipped)
        for p, (xyz, w) in clouds.items():
            np.savez(cloud_files[p], xyz=xyz, weights=w)
    else:
        print("  using cached candidate points (--rebuild 1 to redo segmentation)")

    # 3-5: cluster each prompt and write its records (old records are removed: cluster numbering restarts)
    cluster_dir = args.out_dir + "/clusters"
    os.makedirs(cluster_dir, exist_ok=True)
    counts = {}
    for p in prompts:
        for old in glob.glob(cluster_dir + "/" + prompt_slug(p) + "_[0-9][0-9][0-9][0-9].*"):
            os.remove(old)
        z = np.load(cloud_files[p])
        clusters = cluster_prompt(z['xyz'], z['weights'], args)
        for i, cluster in enumerate(clusters):
            np.random.seed(args.seed)                     # find_pts_in_image samples points randomly
            write_cluster(cluster_dir, p, i, cluster, cluster_views(cluster, params, fList_new, img_size, args))
        counts[p] = len(clusters)
        print(f"  {p}: {len(clusters)} clusters")

    settings = {'format_version': FORMAT_VERSION, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'root_dir': os.path.abspath(args.root_dir), 'nerfacto_dir': os.path.abspath(args.nerfacto_dir),
                'inputs': {'color_dir': COLOR_DIR, 'depth_dir': DEPTH_DIR, 'renders_dir': RENDERS_DIR,
                           'colmap_dir': COLMAP_DIR, 'frame_keyword': FRAME_KEYWORD},
                'out_dir': os.path.abspath(args.out_dir), 'delta_dir': os.path.abspath(args.delta_dir), 'image_size': list(img_size),
                'candidates': args.candidates, 'classifier': args.classifier,
                'sam3_confidence': args.sam3_confidence if args.classifier == 'sam3' else None,
                'threshold': args.threshold,
                'clustering': {'voxel_size': args.voxel_size, 'dbscan_eps': args.dbscan_eps,
                               'dbscan_min_samples': args.dbscan_min_samples, 'min_cluster_voxels': args.min_cluster_voxels},
                'prompts': prompts, 'clusters_per_prompt': counts,
                'box': {'num_points': args.num_points, 'min_points': args.min_points,
                        'box_buffer': args.box_buffer, 'buffer_px': args.buffer_px, 'seed': args.seed},
                'skipped_frames': skipped}
    with open(args.out_dir + "/run_settings.json", 'w') as f:
        json.dump(settings, f, indent=1)
    print(f"wrote {sum(counts.values())} cluster records to {cluster_dir}")