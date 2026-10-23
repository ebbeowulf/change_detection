import argparse
import os
import numpy as np
from PIL import Image


def load_model(model_name, prompts):
    if model_name == 'clipseg':
        from segmentation_utils.clip_segmentation import clip_seg
        return clip_seg(prompts)
    elif model_name == 'yolo-world':
        from segmentation_utils.yolo_world_segmentation import yolo_world_segmentation
        return yolo_world_segmentation(prompts)
    elif model_name == 'dino':
        from segmentation_utils.dino_segmentation import dino_segmentation
        return dino_segmentation(prompts)
    elif model_name == 'sam3':
        from segmentation_utils.sam3_segmentation import sam3_segmentation
        return sam3_segmentation(prompts, threshold=0.1)
    raise ValueError(f"Unknown model: {model_name}")


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def index_by_id(img_dir):
    """Map viewpoint ID -> filename for files named ${PREFIX}_${ID}.{png,jpg,jpeg}.
    The prefix is ignored, so frame_/rgb_/color_ prefixes all work."""
    ids = {}
    for fname in os.listdir(img_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTENSIONS or '_' not in stem:
            continue
        vid = stem.rsplit('_', 1)[1]
        if vid in ids:
            print(f"  WARNING: duplicate ID {vid} in {img_dir} ({ids[vid]}, {fname}); keeping {ids[vid]}")
            continue
        ids[vid] = fname
    return ids


def find_render(render_dir, render_prefix, vid):
    """Return the path of the render ${render_prefix}_${vid} (any image extension), or None if it doesn't exist."""
    for ext in IMAGE_EXTENSIONS:
        path = render_dir + "/" + render_prefix + "_" + vid + ext
        if os.path.exists(path):
            return path
    return None


def same_aspect(size1, size2, tol=0.01):
    return abs(size1[0] / size1[1] - size2[0] / size2[1]) <= tol


def select_color_dir(event, color_dir, rotated_dir, renders_dir, render_prefix):
    """Pick the "after" image directory whose orientation matches the renders.
    Robot cameras are mounted vertically, so robot runs carry a rotated directory alongside color;
    when the color images don't match the renders, the rotated images are used instead.
    Returns (directory name, {id: filename}), or (None, {}) if neither matches."""
    for cand in (color_dir, rotated_dir):
        cdir = event + "/" + cand
        if not os.path.isdir(cdir):
            continue
        ids = index_by_id(cdir)
        for vid in sorted(ids):
            rpath = find_render(event + "/" + renders_dir, render_prefix, vid)
            if rpath is None:
                continue
            if same_aspect(Image.open(rpath).size, Image.open(cdir + "/" + ids[vid]).size):
                return cand, ids
            break  # first matched pair has the wrong orientation; try the next candidate
    return None, {}


def find_events(data_root, color_dir, renders_dir):
    """Any directory under data_root holding both a color and a renders subdirectory is a collection event."""
    events = []
    for dirpath, dirnames, _ in os.walk(data_root):
        if color_dir in dirnames and renders_dir in dirnames:
            events.append(dirpath)
            dirnames[:] = []  # don't descend into the event itself
    return sorted(events)


def get_prob_maps(model, image, num_prompts):
    """Run the segmentation model once and return one probability map per prompt."""
    model.process_image(image)
    probs = []
    for idx in range(num_prompts):
        prob = model.get_prob_array(idx)
        if prob is None:
            prob = np.zeros((image.height, image.width), dtype=np.float32)
        else:
            prob = prob.to('cpu').numpy().astype(np.float32)
        probs.append(prob)
    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build before/after delta masks for every collection event under a data root')
    parser.add_argument('data_root', type=str, help='root directory searched for collection events (dirs containing color and renders subdirs)')
    parser.add_argument('prompts', type=str, nargs='+', help='one or more text queries for the open vocabulary segmentation model')
    parser.add_argument('--model', type=str, default='clipseg', help="segmentation model ('clipseg','yolo-world','dino','sam3') (default=clipseg)")
    parser.add_argument('--color_dir', type=str, default='color', help='name of the captured "after" image subdirectory (default=color)')
    parser.add_argument('--rotated_dir', type=str, default='rotated', help='name of the rotated image subdirectory used when color images do not match the renders in orientation (default=rotated)')
    parser.add_argument('--renders_dir', type=str, default='renders', help='name of the Gaussian Splatting "before" render subdirectory (default=renders)')
    parser.add_argument('--render_prefix', type=str, default='rgb', help='filename prefix of the color renders, i.e. ${render_prefix}_${ID}.png (default=rgb)')
    parser.add_argument('--change_dir', type=str, default='change2.0_results/delta_sam3_0.1', help='name of the output subdirectory; results go in <change_dir>/<model>/<prompt>_<id>.npy (default=delta_change)')
    parser.add_argument('--output_root', type=str, default='', help='optional alternate root for outputs, mirroring the event layout (default = inside each event directory)')
    parser.add_argument('--overwrite', type=int, default=0, help='1 = recompute deltas that already exist on disk (default=0)')
    args = parser.parse_args()

    prompts = args.prompts
    prompt_slugs = [p.strip().lower().replace(' ', '_') for p in prompts]
    model = load_model(args.model, prompts)

    events = find_events(args.data_root, args.color_dir, args.renders_dir)
    print(f"Found {len(events)} collection events under {args.data_root}")

    for event in events:
        rel = os.path.relpath(event, args.data_root)
        after_dir, color_ids = select_color_dir(event, args.color_dir, args.rotated_dir, args.renders_dir, args.render_prefix)
        if after_dir is None:
            print(f"{rel}: WARNING: neither {args.color_dir} nor {args.rotated_dir} matches the renders' orientation - skipping")
            continue
        render_paths = {}
        for vid in color_ids:
            path = find_render(event + "/" + args.renders_dir, args.render_prefix, vid)
            if path is not None:
                render_paths[vid] = path
        common = sorted(render_paths)
        print(f"{rel}: using {after_dir}/, {len(common)} matched pairs ({len(color_ids) - len(common)} images without a render)")

        base_dir = event if args.output_root == '' else args.output_root + "/" + rel
        out_dir = base_dir + "/" + args.change_dir + "/" + args.model
        os.makedirs(out_dir, exist_ok=True)

        warned_size = False
        for vid in common:
            out_files = [out_dir + "/" + slug + "_" + vid + ".npy" for slug in prompt_slugs]
            if not args.overwrite and all(os.path.exists(f) for f in out_files):
                continue

            before = Image.open(render_paths[vid]).convert('RGB')
            after = Image.open(event + "/" + after_dir + "/" + color_ids[vid]).convert('RGB')
            if before.size != after.size:
                if not warned_size:
                    print(f"  NOTE: render size {before.size} != image size {after.size} (same orientation); rescaling renders to match")
                    warned_size = True
                before = before.resize(after.size, Image.BILINEAR)

            probs_before = get_prob_maps(model, before, len(prompts))
            probs_after = get_prob_maps(model, after, len(prompts))
            for idx, out_file in enumerate(out_files):
                delta = probs_after[idx] - probs_before[idx]
                np.save(out_file, delta.astype(np.float16))