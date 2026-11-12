"""Inpaint stage: insert the query object into each planned box with FLUX.1 Fill,
then check with Qwen that the object is really there.

Per plan entry:
  - crop a square context window around the box, scale it to RES, inpaint a
    padded rectangular mask with the entry's prompt, scale back, and paste in only
    the feathered masked region;
  - ask Qwen whether a recognizable query object is visible in that window;
    if not, retry once with a new seed.

Outputs in plan_dir:
  images/<coco_id>_syn.jpg              verified images only
  images/_annotations.coco.json         verified images, category, and the mask box
                                        (the region painted into; loose, not a tight box)
  inpaint_log.json                      settings, prompts, and every attempt with its verdict

Resumable: entries already in the log are skipped.
"""
import argparse
import io
import json
import os
import random
import time

import ollama
import torch
from diffusers import FluxFillPipeline
from PIL import Image, ImageDraw, ImageFilter

from screen_negatives import check_out_dir

# ---------- settings ----------
FLUX_MODEL = "black-forest-labs/FLUX.1-Fill-dev"   # Hugging Face id or local path
VLM_MODEL = "qwen3.5:35b"   # Ollama model tag for the verification check
STEPS = 50                  # denoising steps
GUIDANCE = 30.0             # FLUX guidance
RES = 1024                  # working resolution of the crop sent to FLUX
CONTEXT_MULT = 3.0          # crop side = this times the mask's longest side
MIN_CROP = 384              # smallest crop side, original-image pixels
MASK_PAD = 0.1              # mask = planned box grown by this fraction of its size
FEATHER = 6                 # blur radius (pixels) of the paste-back edge
MAX_TRIES = 2               # inpainting attempts per entry (first + one retry)
JPEG_QUALITY = 95
SEED = 0
PLAN_FILE = "plan.json"
LOG_FILE = "inpaint_log.json"
ANN_FILE = "_annotations.coco.json"
CHECK_MIN = 512             # verification crop is upscaled so its short side is at least this

SETTINGS = dict(FLUX_MODEL=FLUX_MODEL, VLM_MODEL=VLM_MODEL, STEPS=STEPS, GUIDANCE=GUIDANCE, RES=RES,
                CONTEXT_MULT=CONTEXT_MULT, MIN_CROP=MIN_CROP, MASK_PAD=MASK_PAD, FEATHER=FEATHER,
                MAX_TRIES=MAX_TRIES, JPEG_QUALITY=JPEG_QUALITY, SEED=SEED)

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

# SIMPLE_PROMPT = "a {query}, {pose}, on {surface}, photorealistic"   # used for retries
SIMPLE_PROMPT = "a {query}, {pose}, {placement}, photorealistic"

def inpaint_one(pipe, img, box, prompt, seed):
    """Returns (new image, mask_box, crop) without modifying img."""
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

    scale = RES / side
    crop = img.crop((left, top, left + side, top + side)).resize((RES, RES), Image.LANCZOS)
    mask = Image.new("L", (RES, RES), 0)
    ImageDraw.Draw(mask).rectangle([v * scale for v in local], fill=255)

    out = pipe(prompt=prompt, image=crop, mask_image=mask, height=RES, width=RES,
               guidance_scale=GUIDANCE, num_inference_steps=STEPS, max_sequence_length=512,
               generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    # Paste back only the masked region, with soft edges
    paste_mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(paste_mask).rectangle(local, fill=255)
    if FEATHER > 0:
        paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(FEATHER))
    result = img.copy()
    result.paste(out.resize((side, side), Image.LANCZOS), (left, top), paste_mask)
    return result, [round(v, 1) for v in mask_box], [left, top, side]

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
    """Ask Qwen whether the query object is visible in the crop. Returns (present, reason)."""
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
        x1, y1, x2, y2 = inst["mask_box"]
        images.append({"id": iid, "file_name": name, "width": inst["width"], "height": inst["height"],
                       "source_image": inst["image"]})
        anns.append({"id": iid, "image_id": iid, "category_id": category_id,
                     "bbox": [x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)],
                     "area": round((x2 - x1) * (y2 - y1), 1),
                     "iscrowd": 0, "box_source": "inpaint_mask"})
    coco = {"info": {"description": "synthetic '%s' images (inpainted)" % query},
            "images": images, "annotations": anns,
            "categories": [{"id": category_id, "name": query}]}
    with open(path, "w") as f:
        json.dump(coco, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description="Inpaint planned boxes with FLUX.1 Fill and verify with Qwen.")
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
    log = {"query": query, "settings": SETTINGS, "prompts": {"verify": VERIFY_PROMPT, "retry": SIMPLE_PROMPT}, "images": {}}
    if os.path.exists(log_path):
        with open(log_path) as f:
            log["images"] = json.load(f)["images"]

    pipe = FluxFillPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16).to("cuda")
    
    for i, e in enumerate(plan["entries"]):
        if args.num_images and sum(v["verified"] for v in log["images"].values()) >= args.num_images:
            break
        out_name = os.path.basename(e["image"]).rsplit(".", 1)[0] + "_syn.jpg"
        if out_name in log["images"]:
            continue
        t0 = time.time()
        img = Image.open(e["image"]).convert("RGB")
        rng = random.Random(SEED * 1000003 + i)
        attempts = []
        for k in range(MAX_TRIES):
            seed = rng.randint(0, 2**31 - 1)
            prompt = e["prompt"] if k == 0 else SIMPLE_PROMPT.format(
                query=query, pose=e["attributes"]["pose"], placement=e["placement"])
            result, mask_box, crop = inpaint_one(pipe, img, e["box"], prompt, seed)
            present, reason = verify(check_window(result, mask_box), query)
            attempts.append({"seed": seed, "prompt": prompt, "present": present, "reason": reason})
            if present:
                break

        verified = attempts[-1]["present"]
        if verified:
            result.save(img_dir + "/" + out_name, quality=JPEG_QUALITY)
        else:
            os.makedirs(args.plan_dir + "/failed", exist_ok=True)
            result.save(args.plan_dir + "/failed/" + out_name, quality=JPEG_QUALITY)

        log["images"][out_name] = dict(e, mask_box=mask_box, crop=crop, attempts=attempts, verified=verified)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
        write_coco(img_dir + "/" + ANN_FILE, log, query, plan["category_id"])
        print("%d / %d  %s  %s  %.1fs" % (i + 1, len(plan["entries"]), out_name,
                                          "verified" if verified else "FAILED", time.time() - t0))

    done = list(log["images"].values())
    print("%s: %d of %d verified" % (query, sum(v["verified"] for v in done), len(done)))


if __name__ == "__main__":
    main()