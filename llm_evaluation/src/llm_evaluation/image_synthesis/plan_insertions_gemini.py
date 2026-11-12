"""Plan stage, Gemini version: identical to plan_insertions.py except that every
VLM question (attributes, gate, region, size) goes to a Gemini model through the
Google API instead of Qwen through Ollama. Run it with a different --out_dir so
existing plans are never touched.

Choose one placement per image for inpainting the query object.

  0. Screen COCO negatives so none mention the query; skip images with no usable
     reference objects (COCO ground-truth boxes).
  1. Attributes (once per query, text only): Qwen lists attributes and values;
     saved to attributes.json and reused if present.
  2. Gate:   could the query rest somewhere in this scene, and on what surface?
  3. Region: box around that surface.
  4. Size:   with numbered reference objects drawn on the image, the query's height
             relative to one of them and its width-to-height ratio in the chosen pose.
  5. Place:  bottom-center sampled in the lower part of the region, near the
             reference's bottom row, so the size ratio holds.

Outputs in out_dir: screen_report.json, attributes.json, plan.json (settings,
prompts, one entry per planned image, every skipped image with its reason), and
plots/ with one plot per planned image.
"""
import argparse
import base64
import io
import json
import os
import random
import re
import threading
import time
import zlib

from google import genai
from PIL import Image, ImageDraw, ImageFont

from screen_negatives import (REPORT_FILE, check_out_dir, find_images, index_texts,
                              load_coco, query_words, screen_images)

# ---------- settings ----------
MODEL = "gemini-3.5-flash" # Gemini model for all planning questions
THINKING_LEVEL = "low"     # Gemini thinking level: minimal / low / medium / high
API_TIMEOUT = 120          # seconds to wait for one Gemini response
API_WAITS = [10, 30, 60, 120, 300]   # seconds between retries on API errors
TEMPERATURE = 0.2          # VLM sampling temperature
SEED = 0                   # image order, VLM seeds, placement and attribute sampling
MAX_REFS = 5               # reference objects shown for the size question
MIN_REF_FRAC = 0.02        # only size against references covering at least 2% of the image
REGION_LOWER = 0.5         # contact points only in this lower fraction of the region box
BAND_FRAC = 0.1            # contact rows within this fraction of image height of the reference's bottom
MIN_AREA_FRAC = 0.03       # reject boxes smaller than 3% of the image
SIZE_SCALE = 1.5           # empirical correction: box height = SIZE_SCALE * VLM ratio * reference height
MAX_AREA_FRAC = 0.25       # reject boxes larger than this fraction of the image
MAX_OVERLAP = 0.3          # reject boxes with more than this fraction covered by an existing object
PLACE_TRIES = 10           # random placement attempts per image
MAX_ATTEMPT_FACTOR = 30    # give up after num_images * this many images have been sent to the VLM
PLAN_FILE = "plan.json"
ATTR_FILE = "attributes.json"

SETTINGS = dict(MODEL=MODEL, THINKING_LEVEL=THINKING_LEVEL, TEMPERATURE=TEMPERATURE, SEED=SEED, MAX_REFS=MAX_REFS,
                MIN_REF_FRAC=MIN_REF_FRAC, REGION_LOWER=REGION_LOWER, BAND_FRAC=BAND_FRAC,
                MIN_AREA_FRAC=MIN_AREA_FRAC, MAX_AREA_FRAC=MAX_AREA_FRAC, MAX_OVERLAP=MAX_OVERLAP,
                PLACE_TRIES=PLACE_TRIES, MAX_ATTEMPT_FACTOR=MAX_ATTEMPT_FACTOR)

# ---------- prompts ----------
ATTR_PROMPT = (
    "List the visual attributes that vary most between real-world instances of a {query} "
    "in photographs, for building a diverse image dataset. Always include an attribute "
    "named 'pose' whose values are orientations relative to the camera (e.g. 'side view "
    "facing left', 'front view'). Give 3 to 6 attributes in total, each with 4 to 8 short, "
    "distinct values written so they can be inserted into a phrase describing the {query}."
)
GATE_PROMPT = (
    "Could a {query} realistically be added to this scene? It must rest on a visible "
    "surface (such as a floor, ground, road, or table top) with enough open space, at a "
    "size that fits the scene. If yes, set suitable to true and name that surface in a "
    "few words, e.g. 'the concrete floor'. If no, set suitable to false. Give a short "
    "reason either way."
)
REGION_PROMPT = (
    'Detect {surface} in this image. Return JSON as [{{"box_2d": [ymin, xmin, ymax, xmax], '
    '"label": "{surface}"}}], with coordinates normalized to 0-1000.'
)
SIZE_PROMPT = (
    "A {query} ({pose}) will be added to this scene, resting on {surface}. The numbered "
    "yellow boxes mark existing objects: {ref_list}. Choose the numbered object that is "
    "most useful for judging the {query}'s size, preferably one that also rests on "
    "{surface}. Give: ref = that object's number; height_ratio = the {query}'s real-world "
    "height divided by that object's real-world height; aspect = the {query}'s width "
    "divided by its height as it would appear in that pose from this camera's viewpoint; "
    "and a short reason."
)
INPAINT_PROMPT = ("a {query}, {pose} ({details}), on {surface}, photorealistic, blended "
                  "naturally into the surrounding scene with matching lighting, scale and "
                  "camera perspective")
PROMPTS = dict(attributes=ATTR_PROMPT, gate=GATE_PROMPT, region=REGION_PROMPT,
               size=SIZE_PROMPT, inpaint=INPAINT_PROMPT)

ATTR_SCHEMA = {
    "type": "object",
    "properties": {"attributes": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "values": {"type": "array", "items": {"type": "string"}}},
        "required": ["name", "values"]}}},
    "required": ["attributes"],
}
GATE_SCHEMA = {
    "type": "object",
    "properties": {"suitable": {"type": "boolean"}, "surface": {"type": "string"},
                   "reason": {"type": "string"}},
    "required": ["suitable", "surface", "reason"],
}
SIZE_SCHEMA = {
    "type": "object",
    "properties": {"ref": {"type": "integer"}, "height_ratio": {"type": "number"},
                   "aspect": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["ref", "height_ratio", "aspect", "reason"],
}
DEFAULT_POSES = ["side view facing left", "side view facing right", "front view",
                 "rear view", "three-quarter view"]
BBOX_RE = re.compile(r'"box_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]')
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
CLIENT = None   # Gemini client, created in main()


# ---------- VLM ----------

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


def vlm(prompt, image, seed, schema=None):
    """One Gemini call. image is a file path, JPEG bytes, or None (text only).
    With a schema, the reply is asked for as a JSON object with the schema's fields
    and returned as a JSON string. seed is unused (kept so main() matches the Qwen
    version); Gemini sampling is not seeded."""
    if schema is not None:
        prompt += (" Respond with only a JSON object matching this JSON schema: %s"
                   % json.dumps(schema))
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        if isinstance(image, str):
            with open(image, "rb") as f:
                image = f.read()
        content.append({"type": "image", "data": base64.b64encode(image).decode("utf-8"),
                        "mime_type": "image/jpeg"})
    error = ""
    for n, wait in enumerate(API_WAITS):
        try:
            interaction = call_with_timeout(
                CLIENT.interactions.create, API_TIMEOUT, model=MODEL, input=content,
                generation_config={"temperature": TEMPERATURE, "thinking_level": THINKING_LEVEL})
            text = interaction.output_text or ""
            if schema is None:
                return text
            m = JSON_RE.search(text)
            if m is None:
                raise ValueError("no JSON object in reply: " + text[:200])
            return json.dumps(json.loads(m.group(0)))
        except Exception as e:
            error = str(e)[:200]
            if "Error code: 400" in error or isinstance(e, ValueError):
                break
            if n < len(API_WAITS) - 1:
                print("    API error, waiting %ds: %s" % (wait, error), flush=True)
                time.sleep(wait)
    raise RuntimeError("Gemini call failed: " + error)


def pick_region(text, width, height):
    """Largest box_2d ([ymin, xmin, ymax, xmax], 0-1000) in the reply, in pixels and clipped; None if none."""
    best, best_area = None, 0
    for m in BBOX_RE.findall(text):
        y1, x1, y2, x2 = [float(v) for v in m]
        x1, x2 = max(0, x1 * width / 1000), min(width, x2 * width / 1000)
        y1, y2 = max(0, y1 * height / 1000), min(height, y2 * height / 1000)
        area = (x2 - x1) * (y2 - y1)
        if x2 > x1 and y2 > y1 and area > best_area:
            best, best_area = [x1, y1, x2, y2], area
    return best


# ---------- attributes ----------

def load_or_make_attributes(path, query):
    """Reuse attributes.json if present; otherwise ask the VLM (text only) and save it."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    reply = json.loads(vlm(ATTR_PROMPT.format(query=query), None, SEED, ATTR_SCHEMA))
    attrs = {}

    items = reply["attributes"]
    if isinstance(items, dict):   # {"name": [values], ...} instead of a list of objects
        items = [{"name": k, "values": v} for k, v in items.items()]
    for a in items:
        values = [str(v).strip() for v in a["values"] if str(v).strip()]
        if values:
            attrs[a["name"].strip().lower()] = values
    if "pose" not in attrs:
        attrs["pose"] = DEFAULT_POSES
    with open(path, "w") as f:
        json.dump(attrs, f, indent=2)
    return attrs


class Cycler:
    """Cycles through a shuffled copy of each attribute's values, so every value
    is used about equally often."""

    def __init__(self, attrs, rng):
        self.attrs, self.rng = attrs, rng
        self.queues = {k: [] for k in attrs}

    def draw(self, name):
        if not self.queues[name]:
            self.queues[name] = list(self.attrs[name])
            self.rng.shuffle(self.queues[name])
        return self.queues[name].pop()


def make_prompt(query, surface, attributes):
    details = "; ".join("%s: %s" % (k, v) for k, v in attributes.items() if k != "pose")
    return INPAINT_PROMPT.format(query=query, pose=attributes["pose"], details=details, surface=surface)


# ---------- references and drawing ----------

def usable_refs(gt, width, height, edge=2):
    """Largest ground-truth boxes that are big enough and not cut off by the image edge."""
    refs = []
    for r in gt:
        x1, y1, x2, y2 = r["box"]
        if x1 <= edge or y1 <= edge or x2 >= width - edge or y2 >= height - edge:
            continue
        if (x2 - x1) * (y2 - y1) / (width * height) >= MIN_REF_FRAC:
            refs.append(r)
    refs.sort(key=lambda r: (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]), reverse=True)
    return refs[:MAX_REFS]


def get_font(width):
    try:
        return ImageFont.load_default(size=max(14, width // 40))
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def label(draw, xy, text, color, font):
    l, t, r, b = draw.textbbox(xy, text, font=font)
    draw.rectangle([l - 2, t - 2, r + 2, b + 2], fill=(0, 0, 0))
    draw.text(xy, text, fill=color, font=font)


def draw_refs(image_path, refs):
    """JPEG bytes of the image with numbered reference boxes, for the size question."""
    im = Image.open(image_path).convert("RGB")
    draw, font = ImageDraw.Draw(im), get_font(im.size[0])
    for i, r in enumerate(refs, 1):
        draw.rectangle(r["box"], outline=(255, 255, 0), width=3)
        label(draw, (r["box"][0] + 4, r["box"][1] + 4), "%d %s" % (i, r["label"]), (255, 255, 0), font)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def plot_result(image_path, region, ref, box, out_path):
    """Region blue, chosen reference yellow, planned box green."""
    im = Image.open(image_path).convert("RGB")
    draw, font = ImageDraw.Draw(im), get_font(im.size[0])
    draw.rectangle(region, outline=(0, 128, 255), width=3)
    draw.rectangle(ref["box"], outline=(255, 255, 0), width=3)
    label(draw, (ref["box"][0] + 4, ref["box"][1] + 4), "ref " + ref["label"], (255, 255, 0), font)
    draw.rectangle(box, outline=(0, 255, 0), width=3)
    im.save(out_path)


# ---------- placement ----------

def overlap_frac(a, b):
    """Fraction of box a's area covered by box b."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / ((a[2] - a[0]) * (a[3] - a[1]))


def place_box(region, ref_box, box_h, aspect, existing, width, height, prng):
    """Returns (box, None) on success or (None, reason) after PLACE_TRIES attempts."""
    rx1, ry1, rx2, ry2 = region
    y_lo = max(ry2 - REGION_LOWER * (ry2 - ry1), ref_box[3] - BAND_FRAC * height)
    y_hi = min(ry2, ref_box[3] + BAND_FRAC * height)
    if y_hi <= y_lo:
        return None, "no_band_overlap"
    box_w = aspect * box_h
    reason = "no_box_fit"
    for _ in range(PLACE_TRIES):
        cx, cy = prng.uniform(rx1, rx2), prng.uniform(y_lo, y_hi)
        x1, y1, x2, y2 = cx - box_w / 2, cy - box_h, cx + box_w / 2, cy
        if x1 < -0.02 * width or y1 < -0.02 * height or x2 > 1.02 * width:
            reason = "outside"
            continue
        box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
        frac = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
        if not MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC:
            reason = "bad_area"
            continue
        if any(overlap_frac(box, b) > MAX_OVERLAP for b in existing):
            reason = "overlap"
            continue
        return [round(v, 1) for v in box], None
    return None, reason


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Plan one inpainting placement per image with a local VLM.")
    ap.add_argument("--neg_dir", type=str, required=True, help="COCO image directory, e.g. .../train2017")
    ap.add_argument("--ann_dir", type=str, required=True, help="COCO annotations directory")
    ap.add_argument("--query", type=str, required=True, help="Query text, e.g. 'forklift'")
    ap.add_argument("--ignore_words", type=str, default="", help="Comma-separated query words NOT used for screening, e.g. 'red,small'")
    ap.add_argument("--out_dir", type=str, required=True, help="Results folder (must be outside neg_dir)")
    ap.add_argument("--num_images", type=int, required=True, help="Number of images to plan")
    args = ap.parse_args()

    check_out_dir(args.out_dir, args.neg_dir)
    os.makedirs(args.out_dir + "/plots", exist_ok=True)
    global CLIENT
    CLIENT = genai.Client()   # reads GEMINI_API_KEY

    # 0. Screen negatives
    ignore = args.ignore_words.split(",")
    texts, gt, categories = load_coco(args.neg_dir, args.ann_dir)
    kept, report = screen_images(find_images(args.neg_dir), index_texts(texts), args.query, ignore)
    with open(args.out_dir + "/" + REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print("%s: %d negatives kept of %d" % (args.query, report["kept"], report["images_found"]))

    # 1. Attributes
    attrs = load_or_make_attributes(args.out_dir + "/" + ATTR_FILE, args.query)
    cycler = Cycler(attrs, random.Random(SEED + 1))

    order_seed = SEED + zlib.crc32(args.query.encode())   # different image order per category
    rng = random.Random(order_seed)
    rng.shuffle(kept)
    entries, skipped = [], []
    counts = {"vlm_images": 0, "planned": 0, "no_refs": 0, "gate_rejected": 0, "region_fail": 0,
              "size_fail": 0, "no_band_overlap": 0, "outside": 0, "bad_area": 0, "overlap": 0,
              "no_box_fit": 0, "vlm_error": 0}

    def skip(path, stage, detail=""):
        counts[stage] += 1
        skipped.append({"image": path, "stage": stage, "detail": detail})

    for path in kept:
        if counts["planned"] >= args.num_images or counts["vlm_images"] >= MAX_ATTEMPT_FACTOR * args.num_images:
            break
        width, height = Image.open(path).size
        refs = usable_refs(gt.get(path, []), width, height)
        if not refs:
            counts["no_refs"] += 1  # counted only; too many to list
            continue
        counts["vlm_images"] += 1
        vlm_seed = rng.randint(0, 2**31 - 1)

        try:
            # 2. Gate
            gate = json.loads(vlm(GATE_PROMPT.format(query=args.query), path, vlm_seed, GATE_SCHEMA))
            if not gate["suitable"]:
                skip(path, "gate_rejected", gate["reason"])
                continue
            surface = gate["surface"].strip() or "the ground"

            # 3. Region
            region_text = vlm(REGION_PROMPT.format(surface=surface), path, vlm_seed)
            region = pick_region(region_text, width, height)
            if region is None:
                skip(path, "region_fail", region_text[:300])
                continue

            # 4. Size (pose chosen first so the aspect ratio matches it)
            pose = cycler.draw("pose")
            ref_list = ", ".join("%d = %s" % (i, r["label"]) for i, r in enumerate(refs, 1))
            size = json.loads(vlm(SIZE_PROMPT.format(query=args.query, pose=pose, surface=surface, ref_list=ref_list),
                                  draw_refs(path, refs), vlm_seed, SIZE_SCHEMA))
        except Exception as e:
            skip(path, "vlm_error", str(e)[:300])
            continue

        ref_i = int(size["ref"]) - 1
        if not (0 <= ref_i < len(refs)) or size["height_ratio"] <= 0 or not (0.05 <= size["aspect"] <= 20):
            skip(path, "size_fail", json.dumps(size))
            continue
        ref = refs[ref_i]

        # 5. Place
        box_h = SIZE_SCALE * size["height_ratio"] * (ref["box"][3] - ref["box"][1])
        existing = [r["box"] for r in gt.get(path, [])]
        box, reason = place_box(region, ref["box"], box_h, size["aspect"], existing, width, height,
                                random.Random(vlm_seed))
        if box is None:
            skip(path, reason, surface)
            continue

        attributes = {k: cycler.draw(k) for k in attrs if k != "pose"}
        attributes["pose"] = pose
        entries.append({
            "image": path, "width": width, "height": height, "box": box,
            "prompt": make_prompt(args.query, surface, attributes), "attributes": attributes,
            "surface": surface, "region_box": [round(v, 1) for v in region], "ref": ref,
            "height_ratio": size["height_ratio"], "aspect": size["aspect"],
            "gate_reason": gate["reason"], "size_reason": size["reason"], "vlm_seed": vlm_seed,
        })
        plot_result(path, region, ref, box, args.out_dir + "/plots/" + os.path.basename(path))
        counts["planned"] += 1
        print("  planned %d / %d" % (counts["planned"], args.num_images))

    plan = {"query": args.query, "category_id": categories.get(args.query, 1), "order_seed": order_seed,
            "screen_words": sorted(query_words(args.query, ignore)), "settings": SETTINGS,
            "prompts": PROMPTS, "attributes": attrs, "summary": counts,
            "entries": entries, "skipped": skipped}
    with open(args.out_dir + "/" + PLAN_FILE, "w") as f:
        json.dump(plan, f, indent=1)
    print("done:", json.dumps(counts))


if __name__ == "__main__":
    main()