"""Inpaint stage, Gemini automatic-placement version: Gemini chooses where the query
object goes, with no planned box or mask. One Gemini API call per attempt.

For each background in background_ranking.json (best first):
  1. Gate (local Qwen, yes/no): could the query plausibly be present in this scene?
  2. Edit (Gemini, one call): "Add a {query} to this scene where it would naturally
     appear. Keep everything else exactly the same."
  3. Locate: compare original and edited images; the largest region that changed is
     the object's box. Only that region (slightly enlarged, soft-edged) is pasted back
     into the original, so the rest of the photo stays the real image.
  4. Verify (local Qwen): is a recognizable query object visible around that box?
     If not, retry once.

Outputs in cat_dir (which must hold background_ranking.json):
  images/<id>_syn.jpg               verified images only
  images/_annotations.coco.json     verified images, category, box from the difference
  failed/<id>_syn.jpg               last failed attempt, for inspection
  inpaint_log.json                  settings, prompts, gate rejections, every attempt
Needs GEMINI_API_KEY in the environment. Resumable: images already in the log are skipped.
"""
import argparse
import base64
import io
import json
import os
import threading
import time

import numpy as np
import ollama
from google import genai
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from screen_negatives import INST_ANN, check_out_dir, load_oi_classes

# ---------- settings ----------
GEMINI_MODEL = "gemini-nano-banana-2.1"   # Gemini image editing model
IMAGE_SIZE = "1K"           # Gemini output resolution
THINKING_LEVEL = "minimal"  # minimal / medium / high
ASPECTS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]   # output shapes requested from Gemini
API_TIMEOUT = 180           # seconds to wait for one Gemini response
API_WAITS = [10, 30, 60, 120, 300]   # seconds between retries on API errors
VLM_MODEL = "qwen3.5:35b"   # Ollama model tag for gate and verification
MAX_TRIES = 2               # Gemini attempts per accepted background (first + one retry)
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
RANK_FILE = "background_ranking.json"
LOG_FILE = "inpaint_log.json"
ANN_FILE = "_annotations.coco.json"

SETTINGS = dict(GEMINI_MODEL=GEMINI_MODEL, IMAGE_SIZE=IMAGE_SIZE, THINKING_LEVEL=THINKING_LEVEL,
                VLM_MODEL=VLM_MODEL, MAX_TRIES=MAX_TRIES, MAX_GATE_FACTOR=MAX_GATE_FACTOR, BLUR=BLUR,
                DIFF_THRESH=DIFF_THRESH, OPEN_ITERS=OPEN_ITERS, MIN_BOX_FRAC=MIN_BOX_FRAC,
                MAX_BOX_FRAC=MAX_BOX_FRAC, PASTE_PAD=PASTE_PAD, FEATHER=FEATHER,
                CHECK_MIN=CHECK_MIN, JPEG_QUALITY=JPEG_QUALITY)

GATE_PROMPT = ("Could a {query} plausibly be present in this scene without looking out of place? "
               "It does not need to be the most typical place for one.")
GATE_ABSENT = " Answer false if a {query} is already visible in the image."   # Open Images only
GATE_SCHEMA = {"type": "object", "properties": {"suitable": {"type": "boolean"}}, "required": ["suitable"]}
EDIT_PROMPT = ("Add a {query} to this scene where it would naturally appear, at a realistic size. "
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
                       options={"temperature": 0, "seed": 0},
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


# ---------- Gemini ----------

def call_with_timeout(fn, timeout, **kwargs):
    """Run fn(**kwargs) in a background thread; raise TimeoutError if it takes too long."""
    box = {}
    def run():
        try:
            box["value"] = fn(**kwargs)
        except Exception as e:
            box["error"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError("no response after %ds" % timeout)
    if "error" in box:
        raise box["error"]
    return box["value"]


def nearest_aspect(width, height):
    target = width / height
    return min(ASPECTS, key=lambda a: abs(int(a.split(":")[0]) / int(a.split(":")[1]) - target))


def gemini_edit(client, path, size, instruction):
    """One Gemini edit of the image file, retrying on API errors. Returns (image or None, error)."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    content = [{"type": "text", "text": instruction},
               {"type": "image", "data": data, "mime_type": "image/jpeg"}]
    error = ""
    for n, wait in enumerate(API_WAITS):
        try:
            interaction = call_with_timeout(
                client.interactions.create, API_TIMEOUT,
                model=GEMINI_MODEL, input=content,
                response_format={"type": "image", "mime_type": "image/jpeg",
                                 "aspect_ratio": nearest_aspect(*size), "image_size": IMAGE_SIZE},
                generation_config={"thinking_level": THINKING_LEVEL})
            out = interaction.output_image
            if out is None or not out.data:
                return None, "no image returned: " + str(getattr(interaction, "output_text", ""))[:200]
            img = Image.open(io.BytesIO(base64.b64decode(out.data))).convert("RGB")
            return img.resize(size, Image.LANCZOS), None
        except Exception as e:
            error = str(e)[:200]
            if "Error code: 400" in error:   # bad request: retrying will not help
                break
            if n < len(API_WAITS) - 1:
                print("    API error, waiting %ds: %s" % (wait, error), flush=True)
                time.sleep(wait)
    return None, "api_error: " + error


# ---------- locate and paste ----------

def change_box(orig, edited):
    """Box of the largest changed region and the fraction of the image changed outside it.
    Returns (box or None, reason or None, outside_frac)."""
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
    coco = {"info": {"description": "synthetic '%s' images (%s, automatic placement)" % (query, GEMINI_MODEL)},
            "images": images, "annotations": anns,
            "categories": [{"id": category_id, "name": query}]}
    with open(path, "w") as f:
        json.dump(coco, f, indent=1)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Add the query with Gemini (it chooses the place) and verify with Qwen.")
    ap.add_argument("--dataset", type=str, default="coco", choices=["coco", "openimages"], help="Dataset the backgrounds come from (for the category id and gate wording)")
    ap.add_argument("--cat_dir", type=str, required=True, help="Category folder holding background_ranking.json; outputs are written here")
    ap.add_argument("--ann_dir", type=str, required=True, help="Annotations directory (COCO annotations/ or the Open Images folder)")
    ap.add_argument("--num_images", type=int, required=True, help="Stop once this many images are verified")
    args = ap.parse_args()

    with open(args.cat_dir + "/" + RANK_FILE) as f:
        rank = json.load(f)
    query = rank["query"]
    ranked = [r["image"] for r in rank["ranking"]]
    for src_dir in {os.path.dirname(p) for p in ranked}:
        check_out_dir(args.cat_dir, src_dir)
    if args.dataset == "coco":
        with open(args.ann_dir + "/" + INST_ANN) as f:
            categories = {c["name"]: c["id"] for c in json.load(f)["categories"]}
    else:
        categories = {n: i + 1 for i, n in enumerate(load_oi_classes(args.ann_dir).values())}
    category_id = categories.get(query, 1)
    gate_prompt = GATE_PROMPT + (GATE_ABSENT if args.dataset == "openimages" else "")

    img_dir, fail_dir = args.cat_dir + "/images", args.cat_dir + "/failed"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    log_path = args.cat_dir + "/" + LOG_FILE
    log = {"query": query, "dataset": args.dataset, "category_id": category_id, "settings": SETTINGS,
           "prompts": {"gate": gate_prompt, "edit": EDIT_PROMPT, "verify": VERIFY_PROMPT},
           "gate_rejected": [], "images": {}}
    if os.path.exists(log_path):
        with open(log_path) as f:
            old = json.load(f)
        log["gate_rejected"], log["images"] = old["gate_rejected"], old["images"]

    client = genai.Client()   # reads GEMINI_API_KEY
    instruction = EDIT_PROMPT.format(query=query.lower())

    def save_log():
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
        write_coco(img_dir + "/" + ANN_FILE, log, query, category_id)

    for src in ranked:
        verified_count = sum(v["verified"] for v in log["images"].values())
        gate_count = len(log["gate_rejected"]) + len(log["images"])
        if verified_count >= args.num_images or gate_count >= MAX_GATE_FACTOR * args.num_images:
            break
        out_name = os.path.basename(src).rsplit(".", 1)[0] + "_syn.jpg"
        if out_name in log["images"] or src in log["gate_rejected"]:
            continue

        # 1. Gate (local)
        try:
            suitable = bool(ask(gate_prompt.format(query=query.lower()), src, GATE_SCHEMA)["suitable"])
        except Exception as e:
            print("  gate error on %s: %s" % (src, str(e)[:200]))
            continue
        if not suitable:
            log["gate_rejected"].append(src)
            save_log()
            continue

        # 2-4. Edit (one Gemini call), locate, verify
        t0 = time.time()
        orig = Image.open(src).convert("RGB")
        attempts, result = [], orig
        for _ in range(MAX_TRIES):
            edited, error = gemini_edit(client, src, orig.size, instruction)
            if edited is None:
                attempts.append({"present": False, "reason": error})
                continue
            box, reason, outside = change_box(orig, edited)
            if box is None:
                attempts.append({"present": False, "reason": reason, "changed_outside": round(outside, 4)})
                result = edited
                continue
            result = paste_region(orig, edited, box)
            try:
                answer = ask(VERIFY_PROMPT.format(query=query.lower()), check_window(result, box), VERIFY_SCHEMA)
                present, reason = bool(answer["present"]), answer["reason"]
            except Exception as e:
                present, reason = False, "vlm_error: " + str(e)[:200]
            attempts.append({"box": box, "present": present, "reason": reason,
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
    print("%s: %d of %d verified, %d gate rejections, %d Gemini calls"
          % (query, sum(v["verified"] for v in done), len(done), len(log["gate_rejected"]),
             sum(len(v["attempts"]) for v in done)))


if __name__ == "__main__":
    main()