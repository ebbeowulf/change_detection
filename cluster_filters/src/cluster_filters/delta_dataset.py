import glob
import json
import os
import numpy as np
import cv2


def load_split(dataset_root, split_file):
    with open(dataset_root + "/splits/" + split_file) as f:
        return json.load(f)


def load_samples(dataset_root, runs, delta_model, include=(), exclude=()):
    """Collect samples from every label-source file in the given runs ("env/run" names).
    include/exclude are substrings matched against the label-source filename
    (e.g. 'openVocab', 'save_results-orig'). Samples whose delta file doesn't exist are skipped and counted.
    With delta_model=None, delta files are not checked (used for counting labels when building splits)."""
    samples = []
    missing = 0
    for run in runs:
        run_dir = dataset_root + "/" + run
        with open(run_dir + "/meta.json") as f:
            meta = json.load(f)
        delta_dir = meta['source'] + "/" + meta['delta_dir'] + "/" + delta_model if delta_model else None
        for path in sorted(glob.glob(run_dir + "/*.json")):
            fname = os.path.basename(path)
            if fname == 'meta.json':
                continue
            if include and not any(s in fname for s in include):
                continue
            if any(s in fname for s in exclude):
                continue
            with open(path) as f:
                rec = json.load(f)
            for s in rec['samples']:
                delta_path = delta_dir + "/" + rec['prompt'] + "_" + s['frame_id'] + ".npy" if delta_dir else None
                if delta_path is not None and not os.path.exists(delta_path):
                    missing += 1
                    continue
                samples.append({'delta_path': delta_path, 'bbox': s['bbox'], 'label': int(s['changed']),
                                'environment': meta['environment'], 'run': run, 'source_file': fname,
                                'prompt': rec['prompt'],
                                'cluster_id': s['cluster_id'], 'frame_id': s['frame_id']})
    return samples, missing


def sample_key(s):
    return s['source_file'] + "|" + s['cluster_id'] + "|" + s['frame_id']


def square_crop(image, bbox, out_size, scale=1.0, shift=(0.0, 0.0)):
    """Crop a square around bbox [x0,y0,x1,y1] (side = longer box edge * scale), zero-padding
    anything outside the image, then resize to out_size x out_size. Works for HxW and HxWxC arrays."""
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0 + 1, y1 - y0 + 1) * scale
    cx = (x0 + x1) / 2.0 + shift[0] * side
    cy = (y0 + y1) / 2.0 + shift[1] * side
    sx0, sy0 = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    n = max(int(round(side)), 1)

    crop = np.zeros((n, n) + image.shape[2:], dtype=np.float32)
    h, w = image.shape[:2]
    ix0, iy0 = max(sx0, 0), max(sy0, 0)
    ix1, iy1 = min(sx0 + n, w), min(sy0 + n, h)
    if ix1 > ix0 and iy1 > iy0:
        crop[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = image[iy0:iy1, ix0:ix1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


try:
    import torch
    from torch.utils.data import Dataset

    class DeltaCropDataset(Dataset):
        """1-channel delta crops around each cluster box. With augment=True, the box is randomly
        rescaled/shifted and the crop randomly flipped horizontally."""

        def __init__(self, samples, img_size=224, augment=False, jitter_scale=0.1, jitter_shift=0.1):
            self.samples = samples
            self.img_size = img_size
            self.augment = augment
            self.jitter_scale = jitter_scale
            self.jitter_shift = jitter_shift

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            delta = np.load(s['delta_path']).astype(np.float32)
            scale, shift = 1.0, (0.0, 0.0)
            if self.augment:
                scale = 1.0 + np.random.uniform(-self.jitter_scale, self.jitter_scale)
                shift = tuple(np.random.uniform(-self.jitter_shift, self.jitter_shift, size=2))
            crop = square_crop(delta, s['bbox'], self.img_size, scale, shift)
            if self.augment and np.random.rand() < 0.5:
                crop = crop[:, ::-1].copy()
            x = torch.from_numpy(crop).unsqueeze(0)  # (1, H, W)
            y = torch.tensor(float(s['label']))
            return x, y
except ImportError:  # allows load_samples/square_crop to be used without torch installed
    pass