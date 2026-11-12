"""Inpaint stage, Kontext version: let FLUX.1 Kontext choose where the query object
goes, with no planned box. A separate annotation source from the FLUX Fill pipeline.

For each background in background_ranking.json (best first):
  1. Gate (Qwen, yes/no): could the query naturally appear in this scene?
  2. Edit (Kontext): "Add a {query} to this scene in a natural place. Keep everything
     else in the image exactly the same."
  3. Locate: compare original and edited images; the largest region that changed is
     the object's box. Only that region (slightly enlarged, soft-edged) is pasted
     back into the original, so the rest of the photo stays the real COCO image.
  4. Verify (Qwen): is a recognizable query object visible around that box?
     If not, retry once with a new seed.

Outputs in cat_dir (which must hold background_ranking.json):
  images/<coco_id>_syn.jpg          verified images only
  images/_annotations.coco.json     verified images, category, box from the difference
  failed/<coco_id>_syn.jpg          last failed attempt, for inspection
  inpaint_log.json                  settings, prompts, gate rejections, every attempt
Resumable: images already in the log are skipped.
"""
import argparse
import io
import json
import os
import random
import time

import numpy as np
import ollama
import torch
from diffusers import FluxKontextPipeline
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from screen_negatives import INST_ANN, check_out_dir

# ---------- settings ----------
KONTEXT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"   # Hugging Face id or local path
VLM_MODEL = "qwen3.5:35b"   # Ollama model tag for gate and verification
STEPS = 28                  # denoising steps
GUIDANCE = 2.5              # Kontext guidance
MAX_TRIES = 2               # Kontext attempts per accepted background (first + one retry)
MAX_GATE_FACTOR = 30        # stop after num_images * this many gate questions
BLUR = 2                    # blur radius before comparing images (ignores pixel noise)
DIFF_THRESH = 30            # mean RGB difference (0-255) that counts as changed
OPEN_ITERS = 2              # removes small specks from the changed-pixel mask
MIN_BOX_FRAC = 0.002        # changed region smaller than this fraction of the image = nothing added
MAX_BOX_FRAC = 0.6          # changed region larger than this = the whole scene was altered
PASTE_PAD = 0.15            # pasted region = box grown by this fraction of its size
FEATHER = 6                 # blur radius (pixels) of the paste-back edge
CHECK_MIN = 512             # verification crop is upscaled so its short side is at least this
JPEG_QUALITY = 95
SEED = 0
RANK_FILE = "background_ranking.json"
LOG_FILE = "inpaint_log.json"
ANN_FILE = "_annotations.coco.json"

SETTINGS = dict(KONTEXT_MODEL=KONTEXT_MODEL, VLM_MODEL=VLM_MODEL, STEPS=STEPS, GUIDANCE=GUIDANCE,
                MAX_TRIES=MAX_TRIES, MAX_GATE_FACTOR=MAX_GATE_FACTOR, BLUR=BLUR,
                DIFF_THRESH=DIFF_THRESH, OPEN_ITERS=OPEN_ITERS, MIN_BOX_FRAC=MIN_BOX_FRAC,
                MAX_BOX_FRAC=MAX_BOX_FRAC, PASTE_PAD=PASTE_PAD, FEATHER=FEATHER,
                CHECK_MIN=CHECK_MIN, JPEG_QUALITY=JPEG_QUALITY, SEED=SEED)

GATE_PROMPT = "Could a {query} naturally appear in this scene, in the way it is normally found?"
GATE_SCHEMA = {"type": "object", "properties": {"suitable": {"type": "boolean"}}, "required": ["suitable"]}
EDIT_PROMPT = ("Add a {query} to this scene in a natural place. "
               "Keep everything else in the image exactly the same.")
VERIFY_PROMPT = (
    "Is there a clearly recognizable {query} in this image that looks like a natural part "
    "of the scene, not a pasted-in picture, a rectangle of different scenery, or a "
    "distorted shape? Set present to true only if both are true. Give a short reason."
)
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"present": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["present", "reason"],
}


# ---------- Qwen ----------

def ask(prompt, image, schema):
    """One Qwen call on an image path or PIL image; returns the parsed JSON answer."""
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        image = buf.getvalue()
    resp = ollama.chat(model=VLM_MODEL, think=False, format=schema,
                       options={"temperature": 0, "seed": SEED},
                       messages=[{"role": "user", "content": prompt, "images": [image]}])
    return json.loads(resp["message"]["content"])


def check_window(img, box):
    """Crop around the box with a 50% margin, upscaled so its short side is at least CHECK_MIN."""
    width, height = img.size
    x1, y1, x2, y2 = box
    mx, my = 0.5 * (x2 - x1), 0.5 * (y2 - y1)
    win = img.crop((max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)))
    s = CHECK_MIN / min(win.size)
    if s > 1:
        win = win.resize((round(win.size[0] * s), round(win.size[1] * s)), Image.LANCZOS)
    return win


# ---------- edit and locate ----------

def change_box(orig, edited):
    """Box of the largest changed region, and the fraction of the image that changed
    outside it. Returns (box or None, reason or None, outside_frac)."""
    a = np.asarray(orig.filter(ImageFilter.GaussianBlur(BLUR)), dtype=np.float32)
    b = np.asarray(edited.filter(ImageFilter.GaussianBlur(BLUR)), dtype=np.float32)
    changed = np.abs(a - b).mean(axis=2) > DIFF_THRESH
    changed = ndimage.binary_opening(changed, iterations=OPEN_ITERS)
    labels, n = ndimage.label(changed)
    if n == 0:
        return None, "no_change", 0.0
    sizes = ndimage.sum(changed, labels, range(1, n + 1))
    biggest = labels == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(biggest)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    h, w = changed.shape
    outside = float((changed & ~biggest).sum()) / (h * w)
    frac = (box[2] - box[0]) * (box[3] - box[1]) / (h * w)
    if frac < MIN_BOX_FRAC:
        return None, "no_change", outside
    if frac > MAX_BOX_FRAC:
        return None, "global_change", outside
    return box, None, outside


def paste_region(orig, edited, box):
    """Original image with only the (enlarged, feathered) box region taken from the edit."""
    width, height = orig.size
    x1, y1, x2, y2 = box
    dx, dy = PASTE_PAD * (x2 - x1), PASTE_PAD * (y2 - y1)
    mask = Image.new("L", orig.size, 0)
    ImageDraw.Draw(mask).rectangle([max(0, x1 - dx), max(0, y1 - dy),
                                    min(width, x2 + dx), min(height, y2 + dy)], fill=255)
    if FEATHER > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(FEATHER))
    result = orig.copy()
    result.paste(edited, (0, 0), mask)
    return result


def write_coco(path, log, query, category_id):
    """COCO file of verified images; bbox is the changed region found by the difference."""
    images, anns = [], []
    for name, inst in log["images"].items():
        if not inst["verified"]:
            continue
        iid = len(images) + 1
        x1, y1, x2, y2 = inst["box"]
        images.append({"id": iid, "file_name": name, "width": inst["width"], "height": inst["height"],
                       "source_image": inst["image"]})
        anns.append({"id": iid, "image_id": iid, "category_id": category_id,
                     "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                     "iscrowd": 0, "box_source": "difference"})
    coco = {"info": {"description": "synthetic '%s' images (FLUX.1 Kontext)" % query},
            "images": images, "annotations": anns,
            "categories": [{"id": category_id, "name": query}]}
    with open(path, "w") as f:
        json.dump(coco, f, indent=1)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Add the query with FLUX.1 Kontext (no planned box) and verify with Qwen.")
    ap.add_argument("--cat_dir", type=str, required=True, help="Category folder holding background_ranking.json; outputs are written here")
    ap.add_argument("--ann_dir", type=str, required=True, help="COCO annotations directory (for the category id)")
    ap.add_argument("--num_images", type=int, required=True, help="Stop once this many images are verified")
    args = ap.parse_args()

    with open(args.cat_dir + "/" + RANK_FILE) as f:
        rank = json.load(f)
    query = rank["query"]
    ranked = [r["image"] for r in rank["ranking"]]
    for src_dir in {os.path.dirname(p) for p in ranked}:
        check_out_dir(args.cat_dir, src_dir)
    with open(args.ann_dir + "/" + INST_ANN) as f:
        categories = {c["name"]: c["id"] for c in json.load(f)["categories"]}
    category_id = categories.get(query, 1)

    img_dir, fail_dir = args.cat_dir + "/images", args.cat_dir + "/failed"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    log_path = args.cat_dir + "/" + LOG_FILE
    log = {"query": query, "category_id": category_id, "settings": SETTINGS,
           "prompts": {"gate": GATE_PROMPT, "edit": EDIT_PROMPT, "verify": VERIFY_PROMPT},
           "gate_rejected": [], "images": {}}
    if os.path.exists(log_path):
        with open(log_path) as f:
            old = json.load(f)
        log["gate_rejected"], log["images"] = old["gate_rejected"], old["images"]

    pipe = FluxKontextPipeline.from_pretrained(KONTEXT_MODEL, dtype=torch.bfloat16).to("cuda")
    instruction = EDIT_PROMPT.format(query=query)

    def save_log():
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
        write_coco(img_dir + "/" + ANN_FILE, log, query, category_id)

    for i, src in enumerate(ranked):
        verified_count = sum(v["verified"] for v in log["images"].values())
        gate_count = len(log["gate_rejected"]) + len(log["images"])
        if verified_count >= args.num_images or gate_count >= MAX_GATE_FACTOR * args.num_images:
            break
        out_name = os.path.basename(src).rsplit(".", 1)[0] + "_syn.jpg"
        if out_name in log["images"] or src in log["gate_rejected"]:
            continue

        # 1. Gate
        try:
            suitable = bool(ask(GATE_PROMPT.format(query=query), src, GATE_SCHEMA)["suitable"])
        except Exception as e:
            print("  gate error on %s: %s" % (src, str(e)[:200]))
            continue
        if not suitable:
            log["gate_rejected"].append(src)
            save_log()
            continue

        # 2-4. Edit, locate, verify
        t0 = time.time()
        orig = Image.open(src).convert("RGB")
        rng = random.Random(SEED * 1000003 + i)
        attempts, result, box = [], None, None
        for _ in range(MAX_TRIES):
            seed = rng.randint(0, 2**31 - 1)
            edited = pipe(image=orig, prompt=instruction, guidance_scale=GUIDANCE,
                          num_inference_steps=STEPS,
                          generator=torch.Generator("cpu").manual_seed(seed)).images[0]
            edited = edited.convert("RGB").resize(orig.size, Image.LANCZOS)
            box, reason, outside = change_box(orig, edited)
            if box is None:
                attempts.append({"seed": seed, "present": False, "reason": reason, "changed_outside": round(outside, 4)})
                result = edited
                continue
            result = paste_region(orig, edited, box)
            try:
                answer = ask(VERIFY_PROMPT.format(query=query), check_window(result, box), VERIFY_SCHEMA)
                present, reason = bool(answer["present"]), answer["reason"]
            except Exception as e:
                present, reason = False, "vlm_error: " + str(e)[:200]
            attempts.append({"seed": seed, "box": box, "present": present, "reason": reason,
                             "changed_outside": round(outside, 4)})
            if present:
                break

        verified = attempts[-1]["present"]
        result.save((img_dir if verified else fail_dir) + "/" + out_name, quality=JPEG_QUALITY)
        log["images"][out_name] = {"image": src, "width": orig.size[0], "height": orig.size[1],
                                   "box": attempts[-1].get("box"), "attempts": attempts, "verified": verified}
        save_log()
        print("%s  %s  %.1fs  (%d verified)" % (out_name, "verified" if verified else "FAILED",
                                               time.time() - t0, verified_count + int(verified)), flush=True)

    done = list(log["images"].values())
    print("%s: %d of %d verified, %d gate rejections" % (query, sum(v["verified"] for v in done),
                                                         len(done), len(log["gate_rejected"])))


if __name__ == "__main__":
    main()