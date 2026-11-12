"""Inpaint stage, Qwen-Image-Edit version: insert the query object into each planned
box with Qwen-Image-Edit-2511 (instruction editing, no mask input), then check with
Qwen3.5 that the object is really there. Reads the same plan.json as inpaint_boxes.py
(FLUX Fill); run it on a copy of the plan in a separate folder.

Per plan entry:
  - crop a square context window around the box (as in the FLUX version) and draw a
    red outline just outside the area to fill;
  - ask Qwen-Image-Edit to add the object inside the outlined area;
  - find what the editor actually changed near the box and paste back only that region
    (so objects drawn a little larger than the box are not cut off; outline pixels left
    in place are unchanged and never pasted); that region is also the label box;
  - ask Qwen3.5 whether a recognizable query object is visible; if not, retry once
    with the simple prompt.

Outputs in plan_dir:
  images/<id>_syn.jpg               verified images only
  images/_annotations.coco.json     verified images, category, and the changed-region box
  failed/<id>_syn.jpg               last failed attempt, for inspection
  inpaint_log.json                  settings, prompts, every attempt with its verdict
Resumable: entries already in the log are skipped.
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
from diffusers import QwenImageEditPlusPipeline
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from screen_negatives import check_out_dir

# ---------- settings ----------
EDIT_MODEL = "Qwen/Qwen-Image-Edit-2511"   # Hugging Face id or local path
CPU_OFFLOAD = False         # True if the editor and Ollama do not both fit on the GPU (slower)
VLM_MODEL = "qwen3.5:35b"   # Ollama model tag for the verification check
STEPS = 40                  # denoising steps
TRUE_CFG = 4.0              # Qwen-Image-Edit guidance strength
RES = 1024                  # crop is scaled to this before editing
CONTEXT_MULT = 3.0          # crop side = this times the mask's longest side
MIN_CROP = 384              # smallest crop side, original-image pixels
MASK_PAD = 0.1              # mask = planned box grown by this fraction of its size
FEATHER = 6                 # blur radius (pixels) of the paste-back edge
OUTLINE = 4                 # red outline width (pixels at RES), drawn outside the pasted region
SEARCH_MULT = 1.6           # changes are kept within the box grown to this many times its size
DIFF_THRESH = 25            # mean RGB difference (0-255) that counts as changed
BLUR = 2                    # blur radius before comparing (ignores pixel noise)
DILATE = 6                  # grow the changed region by this many pixels (at RES) before pasting
MAX_TRIES = 2               # attempts per entry (first + one retry)
CHECK_MIN = 512             # verification crop is upscaled so its short side is at least this
JPEG_QUALITY = 95
SEED = 0
PLAN_FILE = "plan.json"
LOG_FILE = "inpaint_log.json"
ANN_FILE = "_annotations.coco.json"

SETTINGS = dict(EDIT_MODEL=EDIT_MODEL, CPU_OFFLOAD=CPU_OFFLOAD, VLM_MODEL=VLM_MODEL, STEPS=STEPS,
                TRUE_CFG=TRUE_CFG, RES=RES, CONTEXT_MULT=CONTEXT_MULT, MIN_CROP=MIN_CROP,
                MASK_PAD=MASK_PAD, FEATHER=FEATHER, OUTLINE=OUTLINE, SEARCH_MULT=SEARCH_MULT,
                DIFF_THRESH=DIFF_THRESH, BLUR=BLUR, DILATE=DILATE, MAX_TRIES=MAX_TRIES,
                CHECK_MIN=CHECK_MIN, JPEG_QUALITY=JPEG_QUALITY, SEED=SEED)

EDIT_PROMPT = (
    "Add {object} inside the area outlined in red. The whole object must fit inside the "
    "outline with a little space around it. It must look like part of the original photo, "
    "with matching lighting, shadows, perspective, sharpness and noise. Remove the red "
    "outline and do not change anything else."
)
SIMPLE_PROMPT = "a {query}, {pose}, {placement}, photorealistic"   # object text for retries
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


def inpaint_one(pipe, img, box, instruction, seed):
    """Returns (new image, object box or None, crop) without modifying img.
    The pasted region is what the editor actually changed near the box (so an object
    drawn larger than the box is not cut off), and the object box is that region."""
    width, height = img.size
    x1, y1, x2, y2 = box
    dx, dy = MASK_PAD * (x2 - x1), MASK_PAD * (y2 - y1)
    mask_box = [max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)]

    # Square crop around the mask, kept inside the image
    side = max(CONTEXT_MULT * max(mask_box[2] - mask_box[0], mask_box[3] - mask_box[1]), MIN_CROP)
    side = int(round(min(side, width, height)))
    cx, cy = (mask_box[0] + mask_box[2]) / 2, (mask_box[1] + mask_box[3]) / 2
    left = int(round(min(max(0, cx - side / 2), width - side)))
    top = int(round(min(max(0, cy - side / 2), height - side)))
    local = [mask_box[0] - left, mask_box[1] - top, mask_box[2] - left, mask_box[3] - top]

    # Crop with a red outline just outside the area that will be pasted back
    scale = RES / side
    guide = img.crop((left, top, left + side, top + side)).resize((RES, RES), Image.LANCZOS)
    m = (FEATHER + 2) * scale
    gx1, gy1, gx2, gy2 = [v * scale for v in local]
    ImageDraw.Draw(guide).rectangle([gx1 - m, gy1 - m, gx2 + m, gy2 + m], outline=(255, 0, 0), width=OUTLINE)

    out = pipe(image=[guide], prompt=instruction, negative_prompt=" ", true_cfg_scale=TRUE_CFG,
               num_inference_steps=STEPS,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    out = out.convert("RGB").resize((RES, RES), Image.LANCZOS)

    # What changed, compared with the guide image, near the box (outline pixels the
    # editor left in place are unchanged, so they are never pasted)
    a = np.asarray(guide.filter(ImageFilter.GaussianBlur(BLUR)), dtype=np.float32)
    b = np.asarray(out.filter(ImageFilter.GaussianBlur(BLUR)), dtype=np.float32)
    changed = np.abs(a - b).mean(axis=2) > DIFF_THRESH
    hw, hh = SEARCH_MULT * (gx2 - gx1) / 2, SEARCH_MULT * (gy2 - gy1) / 2
    cxr, cyr = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    allow = np.zeros_like(changed)
    allow[max(0, int(cyr - hh)):int(cyr + hh), max(0, int(cxr - hw)):int(cxr + hw)] = True
    changed = ndimage.binary_opening(changed & allow, iterations=2)
    labels, n = ndimage.label(changed)
    result = img.copy()
    if n == 0:
        return result, None, [left, top, side]
    sizes = ndimage.sum(changed, labels, range(1, n + 1))
    region = labels == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(region)
    obj_box = [round(left + xs.min() / scale, 1), round(top + ys.min() / scale, 1),
               round(left + (xs.max() + 1) / scale, 1), round(top + (ys.max() + 1) / scale, 1)]

    # Paste back only that region, grown a little and soft-edged
    paste = ndimage.binary_dilation(region, iterations=DILATE)
    paste_mask = Image.fromarray((paste * 255).astype(np.uint8)).resize((side, side), Image.BILINEAR)
    if FEATHER > 0:
        paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(FEATHER))
    result.paste(out.resize((side, side), Image.LANCZOS), (left, top), paste_mask)
    return result, obj_box, [left, top, side]


def check_window(img, mask_box):
    """Crop around the mask with a 50% margin, upscaled so its short side is at least CHECK_MIN."""
    width, height = img.size
    x1, y1, x2, y2 = mask_box
    mx, my = 0.5 * (x2 - x1), 0.5 * (y2 - y1)
    win = img.crop((max(0, x1 - mx), max(0, y1 - my), min(width, x2 + mx), min(height, y2 + my)))
    s = CHECK_MIN / min(win.size)
    if s > 1:
        win = win.resize((round(win.size[0] * s), round(win.size[1] * s)), Image.LANCZOS)
    return win


def verify(window, query):
    """Ask Qwen3.5 whether the query object is visible in the crop. Returns (present, reason)."""
    buf = io.BytesIO()
    window.save(buf, format="JPEG", quality=95)
    try:
        resp = ollama.chat(model=VLM_MODEL, think=False, format=VERIFY_SCHEMA,
                           options={"temperature": 0, "seed": SEED},
                           messages=[{"role": "user", "content": VERIFY_PROMPT.format(query=query),
                                      "images": [buf.getvalue()]}])
        answer = json.loads(resp["message"]["content"])
        return bool(answer["present"]), answer["reason"]
    except Exception as e:
        return False, "vlm_error: " + str(e)[:200]


def write_coco(path, log, query, category_id):
    """COCO file of verified images; bbox is the mask box (region painted into)."""
    images, anns = [], []
    for name, inst in log["images"].items():
        if not inst["verified"]:
            continue
        iid = len(images) + 1
        x1, y1, x2, y2 = inst["box_found"]
        images.append({"id": iid, "file_name": name, "width": inst["width"], "height": inst["height"],
                       "source_image": inst["image"]})
        anns.append({"id": iid, "image_id": iid, "category_id": category_id,
                     "bbox": [x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)],
                     "area": round((x2 - x1) * (y2 - y1), 1),
                     "iscrowd": 0, "box_source": "difference"})
    coco = {"info": {"description": "synthetic '%s' images (Qwen-Image-Edit-2511)" % query},
            "images": images, "annotations": anns,
            "categories": [{"id": category_id, "name": query}]}
    with open(path, "w") as f:
        json.dump(coco, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description="Inpaint planned boxes with Qwen-Image-Edit-2511 and verify with Qwen3.5.")
    ap.add_argument("--plan_dir", type=str, required=True, help="Folder holding plan.json; outputs are written here")
    ap.add_argument("--num_images", type=int, default=0, help="Stop once this many images are verified (0 = use all planned images)")
    args = ap.parse_args()

    with open(args.plan_dir + "/" + PLAN_FILE) as f:
        plan = json.load(f)
    query = plan["query"]
    for src_dir in {os.path.dirname(e["image"]) for e in plan["entries"]}:
        check_out_dir(args.plan_dir, src_dir)
    img_dir = args.plan_dir + "/images"
    os.makedirs(img_dir, exist_ok=True)

    log_path = args.plan_dir + "/" + LOG_FILE
    log = {"query": query, "settings": SETTINGS,
           "prompts": {"edit": EDIT_PROMPT, "retry": SIMPLE_PROMPT, "verify": VERIFY_PROMPT}, "images": {}}
    if os.path.exists(log_path):
        with open(log_path) as f:
            log["images"] = json.load(f)["images"]

    pipe = QwenImageEditPlusPipeline.from_pretrained(EDIT_MODEL, dtype=torch.bfloat16)
    if CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for i, e in enumerate(plan["entries"]):
        if args.num_images and sum(v["verified"] for v in log["images"].values()) >= args.num_images:
            break
        out_name = os.path.basename(e["image"]).rsplit(".", 1)[0] + "_syn.jpg"
        if out_name in log["images"]:
            continue
        t0 = time.time()
        img = Image.open(e["image"]).convert("RGB")
        rng = random.Random(SEED * 1000003 + i)
        placement = e.get("placement") or "on " + e.get("surface", "the ground")   # older plans used "surface"
        attempts = []
        for k in range(MAX_TRIES):
            seed = rng.randint(0, 2**31 - 1)
            obj = e["prompt"] if k == 0 else SIMPLE_PROMPT.format(
                query=query, pose=e["attributes"]["pose"], placement=placement)
            result, box_found, crop = inpaint_one(pipe, img, e["box"], EDIT_PROMPT.format(object=obj), seed)
            if box_found is None:
                present, reason = False, "no_change"
            else:
                present, reason = verify(check_window(result, box_found), query)
            attempts.append({"seed": seed, "prompt": obj, "box_found": box_found, "present": present, "reason": reason})
            if present:
                break

        verified = attempts[-1]["present"]
        if verified:
            result.save(img_dir + "/" + out_name, quality=JPEG_QUALITY)
        else:
            os.makedirs(args.plan_dir + "/failed", exist_ok=True)
            result.save(args.plan_dir + "/failed/" + out_name, quality=JPEG_QUALITY)
        log["images"][out_name] = dict(e, box_found=attempts[-1]["box_found"], crop=crop,
                                       attempts=attempts, verified=verified)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
        write_coco(img_dir + "/" + ANN_FILE, log, query, plan["category_id"])
        print("%d / %d  %s  %s  %.1fs" % (i + 1, len(plan["entries"]), out_name,
                                          "verified" if verified else "FAILED", time.time() - t0), flush=True)

    done = list(log["images"].values())
    print("%s: %d of %d verified" % (query, sum(v["verified"] for v in done), len(done)))


if __name__ == "__main__":
    main()