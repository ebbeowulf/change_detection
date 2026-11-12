"""Inpaint stage, Gemini version: insert the query object into each planned box with
Gemini's image editing model (Nano Banana 2.1), then check with Qwen that the object
is really there. Same plan.json input and same check as inpaint_boxes.py (FLUX), with
separate output names so the two engines can be compared on the same plans.

Per plan entry:
  - crop a square context window around the box (as in the FLUX version);
  - send Gemini two images: the crop, and a copy with the target area outlined in red,
    plus an instruction to add the object inside that area of the first image
    (Gemini takes no mask file, so the outline marks the area);
  - scale the returned image back and paste in only the feathered mask region, so
    nothing outside the box can change;
  - ask Qwen whether a recognizable query object is visible; if not, retry once with
    the simple prompt.

Outputs in plan_dir:
  images_gemini/<coco_id>_syn.jpg           verified images only
  images_gemini/_annotations.coco.json      verified images, category, and the mask box
  failed_gemini/<coco_id>_syn.jpg           last failed attempt, for inspection
  inpaint_log_gemini.json                   settings, prompts, every attempt with its verdict

Needs GEMINI_API_KEY in the environment (a billed project: image models are not on
the free tier). Resumable: entries already in the log are skipped.
"""
import argparse
import base64
import io
import json
import os
import time

import ollama
from google import genai
from PIL import Image, ImageDraw, ImageFilter

from screen_negatives import check_out_dir

# ---------- settings ----------
GEMINI_MODEL = "gemini-nano-banana-2.1"   # Gemini image editing model
IMAGE_SIZE = "1K"           # Gemini output resolution (square, 1:1)
THINKING_LEVEL = "medium"   # minimal / medium / high (medium is the model default)
VLM_MODEL = "qwen3.5:35b"   # Ollama model tag for the verification check
RES = 1024                  # crop is scaled to this before sending
CONTEXT_MULT = 3.0          # crop side = this times the mask's longest side
MIN_CROP = 384              # smallest crop side, original-image pixels
MASK_PAD = 0.1              # mask = planned box grown by this fraction of its size
FEATHER = 6                 # blur radius (pixels) of the paste-back edge
OUTLINE = 4                 # red outline width in the guide image (pixels at RES)
MAX_TRIES = 2               # attempts per entry (first + one retry)
API_WAITS = [10, 30, 60, 120, 300]   # seconds to wait between API retries on errors
JPEG_QUALITY = 95
CHECK_MIN = 512             # verification crop is upscaled so its short side is at least this
PLAN_FILE = "plan.json"
LOG_FILE = "inpaint_log_gemini.json"
IMG_SUBDIR = "images_gemini"
FAIL_SUBDIR = "failed_gemini"
ANN_FILE = "_annotations.coco.json"

SETTINGS = dict(GEMINI_MODEL=GEMINI_MODEL, IMAGE_SIZE=IMAGE_SIZE, THINKING_LEVEL=THINKING_LEVEL,
                VLM_MODEL=VLM_MODEL, RES=RES, CONTEXT_MULT=CONTEXT_MULT, MIN_CROP=MIN_CROP,
                MASK_PAD=MASK_PAD, FEATHER=FEATHER, OUTLINE=OUTLINE, MAX_TRIES=MAX_TRIES,
                JPEG_QUALITY=JPEG_QUALITY, CHECK_MIN=CHECK_MIN)

EDIT_PROMPT = (
    "Edit the first image. The second image shows the same scene with a red rectangle "
    "marking an area. Add {object} inside that area of the first image, filling most of "
    "it. It must look like it was in the original photo, with matching lighting, shadows, "
    "camera angle, sharpness and noise. Do not change anything outside that area, and do "
    "not add any red marks, borders or text. Return the edited first image with the same "
    "framing."
)
SIMPLE_PROMPT = "a {query}, {pose}, on {surface}, photorealistic"   # object text for retries
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


def to_b64(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def gemini_edit(client, crop, guide, instruction):
    """One Gemini edit, retrying on API errors. Returns (image or None, error or None)."""
    content = [{"type": "text", "text": instruction},
               {"type": "image", "data": to_b64(crop), "mime_type": "image/jpeg"},
               {"type": "image", "data": to_b64(guide), "mime_type": "image/jpeg"}]
    error = ""
    for n, wait in enumerate(API_WAITS):
        try:
            interaction = client.interactions.create(
                model=GEMINI_MODEL,
                input=content,
                response_format={"type": "image", "mime_type": "image/jpeg",
                                 "aspect_ratio": "1:1", "image_size": IMAGE_SIZE},
                generation_config={"thinking_level": THINKING_LEVEL},
            )
            out = interaction.output_image
            if out is None or not out.data:
                return None, "no image returned: " + str(getattr(interaction, "output_text", ""))[:200]
            return Image.open(io.BytesIO(base64.b64decode(out.data))).convert("RGB"), None
        except Exception as e:
            error = str(e)[:200]
            if "Error code: 400" in error:   # bad request: retrying will not help
                break
            if n < len(API_WAITS) - 1:                
                print("    API error, waiting %ds: %s" % (wait, error))
                time.sleep(wait)
    return None, "api_error: " + error


def inpaint_one(client, img, box, instruction):
    """Returns (new image or None, mask_box, crop, error) without modifying img."""
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
    crop_info = [left, top, side]

    scale = RES / side
    crop = img.crop((left, top, left + side, top + side)).resize((RES, RES), Image.LANCZOS)
    guide = crop.copy()
    ImageDraw.Draw(guide).rectangle([v * scale for v in local], outline=(255, 0, 0), width=OUTLINE)

    out, error = gemini_edit(client, crop, guide, instruction)
    if out is None:
        return None, [round(v, 1) for v in mask_box], crop_info, error

    # Paste back only the masked region, with soft edges
    paste_mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(paste_mask).rectangle(local, fill=255)
    if FEATHER > 0:
        paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(FEATHER))
    result = img.copy()
    result.paste(out.resize((side, side), Image.LANCZOS), (left, top), paste_mask)
    return result, [round(v, 1) for v in mask_box], crop_info, None


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
                           options={"temperature": 0, "seed": 0},
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
    coco = {"info": {"description": "synthetic '%s' images (inpainted with %s)" % (query, GEMINI_MODEL)},
            "images": images, "annotations": anns,
            "categories": [{"id": category_id, "name": query}]}
    with open(path, "w") as f:
        json.dump(coco, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description="Inpaint planned boxes with Gemini and verify with Qwen.")
    ap.add_argument("--plan_dir", type=str, required=True, help="Folder holding plan.json; outputs are written here")
    ap.add_argument("--num_images", type=int, default=0, help="Stop once this many images are verified (0 = use all planned images)")
    args = ap.parse_args()

    with open(args.plan_dir + "/" + PLAN_FILE) as f:
        plan = json.load(f)
    query = plan["query"]
    for src_dir in {os.path.dirname(e["image"]) for e in plan["entries"]}:
        check_out_dir(args.plan_dir, src_dir)
    img_dir = args.plan_dir + "/" + IMG_SUBDIR
    os.makedirs(img_dir, exist_ok=True)

    log_path = args.plan_dir + "/" + LOG_FILE
    log = {"query": query, "settings": SETTINGS,
           "prompts": {"edit": EDIT_PROMPT, "retry": SIMPLE_PROMPT, "verify": VERIFY_PROMPT}, "images": {}}
    if os.path.exists(log_path):
        with open(log_path) as f:
            log["images"] = json.load(f)["images"]

    print("Opening genai client")
    client = genai.Client()   # reads GEMINI_API_KEY

    for i, e in enumerate(plan["entries"]):
        if args.num_images and sum(v["verified"] for v in log["images"].values()) >= args.num_images:
            break
        out_name = os.path.basename(e["image"]).rsplit(".", 1)[0] + "_syn.jpg"
        if out_name in log["images"]:
            continue
        t0 = time.time()
        img = Image.open(e["image"]).convert("RGB")
        pose = e.get("pose") or e["attributes"]["pose"]
        attempts, result = [], None
        for k in range(MAX_TRIES):
            obj = e["prompt"] if k == 0 else SIMPLE_PROMPT.format(query=query, pose=pose, surface=e["surface"])
            instruction = EDIT_PROMPT.format(object=obj)
            out, mask_box, crop, error = inpaint_one(client, img, e["box"], instruction)
            if out is None:
                attempts.append({"prompt": obj, "present": False, "reason": error})
                continue
            result = out
            present, reason = verify(check_window(result, mask_box), query)
            attempts.append({"prompt": obj, "present": present, "reason": reason})
            if present:
                break

        verified = attempts[-1]["present"]
        if verified:
            result.save(img_dir + "/" + out_name, quality=JPEG_QUALITY)
        elif result is not None:
            os.makedirs(args.plan_dir + "/" + FAIL_SUBDIR, exist_ok=True)
            result.save(args.plan_dir + "/" + FAIL_SUBDIR + "/" + out_name, quality=JPEG_QUALITY)
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