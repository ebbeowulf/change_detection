import json
import os
import numpy as np
from PIL import Image
from build_delta_change_masks import index_by_id, find_render
from delta_dataset import square_crop, sample_key

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 14


def add_image_paths(dataset_root, samples):
    """Set s['after_path'] and s['before_path'] from each run's meta.json. Returns the samples that have both."""
    runs = {}
    kept = []
    for s in samples:
        if s['run'] not in runs:
            with open(dataset_root + "/" + s['run'] + "/meta.json") as f:
                meta = json.load(f)
            runs[s['run']] = (meta, index_by_id(meta['source'] + "/" + meta['color_dir']))
        meta, ids = runs[s['run']]
        if s['frame_id'] not in ids:
            continue
        before = find_render(meta['source'] + "/" + meta['renders_dir'], meta['render_prefix'], s['frame_id'])
        if before is None:
            continue
        s['after_path'] = meta['source'] + "/" + meta['color_dir'] + "/" + ids[s['frame_id']]
        s['before_path'] = before
        kept.append(s)
    return kept


def cache_tag(model_name, img_size, gray):
    return model_name.split('/')[-1] + "_" + str(img_size) + ("_gray" if gray else "_rgb")


def to_input(crop, gray):
    """HxWx3 float crop in 0-255 -> normalized 3xHxW; gray converts to luminance copied to 3 channels."""
    x = crop / 255.0
    if gray:
        g = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        x = np.repeat(g[..., None], 3, axis=2)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)


def delta_patch_map(delta_crop, img_size):
    """Mean positive delta in each DINOv2 patch -> (img_size/14)^2 values."""
    g = img_size // PATCH
    d = np.clip(delta_crop, 0, None)[:g * PATCH, :g * PATCH]
    return d.reshape(g, PATCH, g, PATCH).mean(axis=(1, 3)).reshape(-1).astype(np.float32)


def _load_crops(s, img_size, gray, cache):
    """Before/after/delta crops for one sample. cache holds the most recently decoded full images,
    since consecutive samples (sorted by frame) usually share them."""
    def get(path, loader):
        if path not in cache:
            if len(cache) >= 6:
                cache.pop(next(iter(cache)))
            cache[path] = loader(path)
        return cache[path]

    a_img = get(s['after_path'], lambda p: np.asarray(Image.open(p).convert('RGB'), dtype=np.float32))
    b_img = get(s['before_path'], lambda p: Image.open(p).convert('RGB'))
    if b_img.size != (a_img.shape[1], a_img.shape[0]):
        b_img = b_img.resize((a_img.shape[1], a_img.shape[0]), Image.BILINEAR)
    b_arr = np.asarray(b_img, dtype=np.float32)
    delta = get(s['delta_path'], lambda p: np.load(p).astype(np.float32))
    return (to_input(square_crop(b_arr, s['bbox'], img_size), gray),
            to_input(square_crop(a_img, s['bbox'], img_size), gray),
            delta_patch_map(square_crop(delta, s['bbox'], img_size), img_size))


def extract_features(dataset_root, samples, model_name, img_size, gray, batch_size, device, num_workers=16):
    """Run frozen DINOv2 on the before (render) and after crops of every sample and cache
    the full token arrays (CLS + patches) with the per-patch delta map in
    <dataset_root>/<run>/dino_cache/<tag>/<sample>.npz. Only uncached samples are computed.
    Image decoding and cropping run in num_workers DataLoader workers so the GPU isn't left waiting."""
    if img_size % PATCH:
        raise ValueError(f"img_size must be a multiple of {PATCH}")
    tag = cache_tag(model_name, img_size, gray)
    for s in samples:
        s['feat_path'] = (dataset_root + "/" + s['run'] + "/dino_cache/" + tag + "/"
                          + sample_key(s).replace("|", "__") + ".npz")
    todo = [s for s in samples if not os.path.exists(s['feat_path'])]
    if not todo:
        return
    print(f"  extracting {model_name} features for {len(todo)} samples ({len(samples) - len(todo)} cached)")
    # neighbouring samples share frames, so each worker's small image cache gets reused within a batch
    todo.sort(key=lambda s: (s['after_path'], s['delta_path']))

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel

    class CropDataset(Dataset):
        def __init__(self):
            self.cache = {}

        def __len__(self):
            return len(todo)

        def __getitem__(self, idx):
            b, a, d = _load_crops(todo[idx], img_size, gray, self.cache)
            return torch.from_numpy(b), torch.from_numpy(a), torch.from_numpy(d), idx

    model = AutoModel.from_pretrained(model_name).to(device).eval()
    loader = DataLoader(CropDataset(), batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=(device == 'cuda'))
    done = 0
    for b, a, d, idx in loader:
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            tokens = model(pixel_values=torch.cat([b, a]).to(device, non_blocking=True)).last_hidden_state
        tokens = tokens.float().cpu().numpy().astype(np.float16)
        n = len(idx)
        for i, j in enumerate(idx.tolist()):
            s = todo[j]
            os.makedirs(os.path.dirname(s['feat_path']), exist_ok=True)
            tmp = s['feat_path'][:-4] + ".tmp.npz"
            np.savez(tmp, before=tokens[i], after=tokens[n + i], delta=d[i].numpy())
            os.replace(tmp, s['feat_path'])  # atomic, so an interrupted run never leaves a partial cache file
        done += n
        print(f"    {done}/{len(todo)}", end="\r")
    print()