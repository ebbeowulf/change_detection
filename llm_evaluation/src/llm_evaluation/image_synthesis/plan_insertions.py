"""Plan stage: choose one placement per image for inpainting the query object.

  0. Screen COCO negatives so none mention the query; skip images with no usable
     reference objects (COCO ground-truth boxes).
  1. Attributes (once per query, text only): Qwen lists attributes and values;
     saved to attributes.json and reused if present.
  2. Gate:   could the query naturally appear in this scene? If so, where (placement
             phrase), on/in/attached to what (anchor), and does it rest on top of it?
  3. Region: box around the anchor.
  4. Size:   with numbered reference objects drawn on the image, the query's height
             relative to one of them and its width-to-height ratio in the chosen pose.
  5. Place:  if it rests on the anchor, bottom-center sampled in the lower part of the
             region near the reference's bottom row, so the size ratio holds;
             otherwise box center sampled anywhere inside the region.
  Backgrounds are tried in the order of background_ranking.json (rank_backgrounds.py)
  if that file is in out_dir, otherwise in a random order per category.

Outputs in out_dir: screen_report.json, attributes.json, plan.json (settings,
prompts, one entry per planned image, every skipped image with its reason), and
plots/ with one plot per planned image.
"""
import argparse
import io
import json
import os
import random
import re
import zlib

import ollama
from PIL import Image, ImageDraw, ImageFont

from screen_negatives import (REPORT_FILE, check_out_dir, find_images, index_texts,
                              load_dataset, query_words, related_names, screen_images)

# ---------- settings ----------
MODEL = "qwen3.5:35b"      # Ollama model tag
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
RANK_FILE = "background_ranking.json"   # from rank_backgrounds.py; used if present in out_dir
ATTR_FILE = "attributes.json"

SETTINGS = dict(MODEL=MODEL, TEMPERATURE=TEMPERATURE, SEED=SEED, MAX_REFS=MAX_REFS,
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
    "Could a {query} plausibly be present in this scene without looking out of place? "
    "It does not need to be the most typical place for one. "    
    "If yes, set suitable to true and give: placement = a short phrase saying where and "
    "how it would appear in this scene; anchor = the visible part of the scene it would "
    "be on, in, or attached to, in a few words; rests_on = true if it would sit on top "
    "of the anchor, false otherwise. If no, set suitable to false."
)
REGION_PROMPT = (
    'Locate {anchor} in this image. Report its bounding box in JSON format as '
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{anchor}"}}].'
)
SIZE_PROMPT = (
    "A {query} ({pose}) will be added to this scene, {placement}. The numbered yellow "
    "boxes mark existing objects: {ref_list}. Choose the numbered object that is most "
    "useful for judging the {query}'s size, preferably one at a similar distance from the "
    "camera. Give: ref = that object's number; height_ratio = the {query}'s real-world "
    "height divided by that object's real-world height; aspect = the {query}'s width "
    "divided by its height as it would appear in that pose from this camera's viewpoint."
)
INPAINT_PROMPT = ("a {query}, {pose} ({details}), {placement}, photorealistic, blended "
                  "naturally into the surrounding scene with matching lighting, scale and "
                  "camera perspective")
GATE_ABSENT = " Set suitable to false if a {query} is already visible in the image."   # Open Images only:
# its labels are not exhaustive, so the gate also checks that the object is not already there
PROMPTS = dict(gate_absent_openimages=GATE_ABSENT, attributes=ATTR_PROMPT, gate=GATE_PROMPT, region=REGION_PROMPT,
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
    "properties": {"suitable": {"type": "boolean"}, "placement": {"type": "string"},
                   "anchor": {"type": "string"}, "rests_on": {"type": "boolean"}},
    "required": ["suitable", "placement", "anchor", "rests_on"],
}
SIZE_SCHEMA = {
    "type": "object",
    "properties": {"ref": {"type": "integer"}, "height_ratio": {"type": "number"},
                   "aspect": {"type": "number"}},
    "required": ["ref", "height_ratio", "aspect"],
}
DEFAULT_POSES = ["side view facing left", "side view facing right", "front view",
                 "rear view", "three-quarter view"]
BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]')


# ---------- VLM ----------

def vlm(prompt, image, seed, schema=None):
    """One Ollama call. image is a file path, JPEG bytes, or None (text only)."""
    msg = {"role": "user", "content": prompt}
    if image is not None:
        msg["images"] = [image]
    kwargs = {"format": schema} if schema is not None else {}
    resp = ollama.chat(model=MODEL, messages=[msg], think=False,
                       options={"temperature": TEMPERATURE, "seed": seed}, **kwargs)
    return resp["message"]["content"]


def pick_region(text, width, height):
    """Largest bbox_2d (0-1000 coordinates) in the reply, in pixels and clipped; None if none."""
    best, best_area = None, 0
    for m in BBOX_RE.findall(text):
        x1, y1, x2, y2 = [float(v) for v in m]
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
    for a in reply["attributes"]:
        values = [v.strip() for v in a["values"] if v.strip()]
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


def make_prompt(query, placement, attributes):
    details = "; ".join("%s: %s" % (k, v) for k, v in attributes.items() if k != "pose")
    return INPAINT_PROMPT.format(query=query, pose=attributes["pose"], details=details, placement=placement)


# ---------- references and drawing ----------

def pixel_boxes(gt, width, height):
    """Ground-truth boxes in pixels (Open Images boxes are stored as fractions)."""
    out = []
    for r in gt:
        b = r["box"]
        if r.get("norm"):
            b = [b[0] * width, b[1] * height, b[2] * width, b[3] * height]
        out.append({"label": r["label"], "box": b})
    return out


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


def place_box(region, ref_box, box_h, aspect, rests_on, existing, width, height, prng):
    """Returns (box, None) on success or (None, reason) after PLACE_TRIES attempts.
    rests_on: bottom-center in the lower part of the region, near the reference's
    bottom row. Otherwise: box center anywhere inside the region."""
    rx1, ry1, rx2, ry2 = region
    if rests_on:
        y_lo = max(ry2 - REGION_LOWER * (ry2 - ry1), ref_box[3] - BAND_FRAC * height)
        y_hi = min(ry2, ref_box[3] + BAND_FRAC * height)
        if y_hi <= y_lo:
            return None, "no_band_overlap"
    else:
        y_lo, y_hi = ry1, ry2
    box_w = aspect * box_h
    reason = "no_box_fit"
    for _ in range(PLACE_TRIES):
        cx, cy = prng.uniform(rx1, rx2), prng.uniform(y_lo, y_hi)
        if rests_on:
            x1, y1, x2, y2 = cx - box_w / 2, cy - box_h, cx + box_w / 2, cy
        else:
            x1, y1, x2, y2 = cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2
        if x1 < -0.02 * width or y1 < -0.02 * height or x2 > 1.02 * width or y2 > 1.02 * height:
            reason = "outside"
            continue
        box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
        frac = (box[2] - box[0]) * (box[3] - box[1]) / (width * height)
        if not MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC:
            reason = "bad_area"
            continue
        if rests_on and any(overlap_frac(box, b) > MAX_OVERLAP for b in existing):   # attached objects may overlap their anchor
            reason = "overlap"
            continue
        return [round(v, 1) for v in box], None
    return None, reason


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Plan one inpainting placement per image with a local VLM.")
    ap.add_argument("--dataset", type=str, default="coco", choices=["coco", "openimages"], help="Which dataset's annotations to use")
    ap.add_argument("--neg_dir", type=str, required=True, help="Image directory, e.g. .../train2017 or .../openimages/validation")
    ap.add_argument("--ann_dir", type=str, required=True, help="Annotations directory (COCO annotations/ or the Open Images folder)")
    ap.add_argument("--query", type=str, required=True, help="Query text, e.g. 'forklift'")
    ap.add_argument("--ignore_words", type=str, default="", help="Comma-separated query words NOT used for screening, e.g. 'red,small'")
    ap.add_argument("--out_dir", type=str, required=True, help="Results folder (must be outside neg_dir)")
    ap.add_argument("--num_images", type=int, required=True, help="Number of images to plan")
    args = ap.parse_args()

    check_out_dir(args.out_dir, args.neg_dir)
    os.makedirs(args.out_dir + "/plots", exist_ok=True)

    # 0. Screen negatives
    ignore = args.ignore_words.split(",")
    texts, gt, categories = load_dataset(args.dataset, args.neg_dir, args.ann_dir)
    related = related_names(args.dataset, args.ann_dir, args.query)
    kept, report = screen_images(find_images(args.neg_dir), index_texts(texts), args.query, ignore, related)
    gate_prompt = GATE_PROMPT + (GATE_ABSENT if args.dataset == "openimages" else "")
    with open(args.out_dir + "/" + REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print("%s: %d negatives kept of %d" % (args.query, report["kept"], report["images_found"]))

    # 1. Attributes
    attrs = load_or_make_attributes(args.out_dir + "/" + ATTR_FILE, args.query)

    order_seed = SEED + zlib.crc32(args.query.encode())   # different image order per category
    rng = random.Random(order_seed)
    rank_path = args.out_dir + "/" + RANK_FILE
    if os.path.exists(rank_path):   # try backgrounds in similarity order (rank_backgrounds.py)
        with open(rank_path) as f:
            ranked = [r["image"] for r in json.load(f)["ranking"]]
        kept_set = set(kept)
        kept = [p for p in ranked if p in kept_set]
        background_order = "ranked"
    else:
        rng.shuffle(kept)
        background_order = "random"
    print("background order: %s, %d candidates" % (background_order, len(kept)))
    entries, skipped = [], []
    counts = {"vlm_images": 0, "planned": 0, "no_refs": 0, "gate_rejected": 0, "region_fail": 0,
              "size_fail": 0, "no_band_overlap": 0, "outside": 0, "bad_area": 0, "overlap": 0,
              "no_box_fit": 0, "vlm_error": 0}
    plan_path = args.out_dir + "/" + PLAN_FILE
    if os.path.exists(plan_path):   # continue an earlier plan: keep its results, never re-ask about them
        with open(plan_path) as f:
            old = json.load(f)
        entries = old["entries"]
        skipped = [s for s in old["skipped"] if s["stage"] != "vlm_error"]   # errors are retried
        for k in counts:
            counts[k] = old["summary"].get(k, 0)
        counts["vlm_images"] -= counts["vlm_error"]
        counts["vlm_error"] = 0
        counts["no_refs"] = 0   # recounted below (cheap, no VLM call)
        print("continuing plan.json: %d planned, %d skipped" % (len(entries), len(skipped)))
    done = {e["image"] for e in entries} | {s["image"] for s in skipped}
    cycler = Cycler(attrs, random.Random(SEED + 1 + len(entries)))
    start_planned, start_vlm = counts["planned"], counts["vlm_images"]

    def skip(path, stage, detail=""):
        counts[stage] += 1
        skipped.append({"image": path, "stage": stage, "detail": detail})

    for path in kept:
        if counts["planned"] >= args.num_images or \
                counts["vlm_images"] - start_vlm >= MAX_ATTEMPT_FACTOR * (args.num_images - start_planned):
            break   # attempt limit applies to this round's new images only
        if path in done:
            continue
        width, height = Image.open(path).size
        gt_px = pixel_boxes(gt.get(path, []), width, height)
        refs = usable_refs(gt_px, width, height)
        if not refs:
            counts["no_refs"] += 1  # counted only; too many to list
            continue
        counts["vlm_images"] += 1
        vlm_seed = (SEED + zlib.crc32(path.encode())) % 2**31   # per image, the same in every round

        try:
            # 2. Gate
            gate = json.loads(vlm(gate_prompt.format(query=args.query), path, vlm_seed, GATE_SCHEMA))
            placement, anchor = gate["placement"].strip(), gate["anchor"].strip()
            if not gate["suitable"] or not placement or not anchor:
                skip(path, "gate_rejected")
                continue

            # 3. Region
            region_text = vlm(REGION_PROMPT.format(anchor=anchor), path, vlm_seed)
            region = pick_region(region_text, width, height)
            if region is None:
                skip(path, "region_fail", region_text[:300])
                continue

            # 4. Size (pose chosen first so the aspect ratio matches it)
            pose = cycler.draw("pose")
            ref_list = ", ".join("%d = %s" % (i, r["label"]) for i, r in enumerate(refs, 1))
            size = json.loads(vlm(SIZE_PROMPT.format(query=args.query, pose=pose, placement=placement, ref_list=ref_list),
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
        existing = [r["box"] for r in gt_px]
        box, reason = place_box(region, ref["box"], box_h, size["aspect"], bool(gate["rests_on"]),
                                existing, width, height, random.Random(vlm_seed))
        if box is None:
            skip(path, reason, placement)
            continue

        attributes = {k: cycler.draw(k) for k in attrs if k != "pose"}
        attributes["pose"] = pose
        entries.append({
            "image": path, "width": width, "height": height, "box": box,
            "prompt": make_prompt(args.query, placement, attributes), "attributes": attributes,
            "placement": placement, "anchor": anchor, "rests_on": bool(gate["rests_on"]),
            "region_box": [round(v, 1) for v in region], "ref": ref,
            "height_ratio": size["height_ratio"], "aspect": size["aspect"], "vlm_seed": vlm_seed,
        })
        plot_result(path, region, ref, box, args.out_dir + "/plots/" + os.path.basename(path))
        counts["planned"] += 1
        print("  planned %d / %d" % (counts["planned"], args.num_images))

    plan = {"query": args.query, "dataset": args.dataset, "related_names": related,
            "category_id": categories.get(args.query, 1), "order_seed": order_seed, "background_order": background_order,
            "screen_words": sorted(query_words(args.query, ignore)), "settings": SETTINGS,
            "prompts": PROMPTS, "attributes": attrs, "summary": counts,
            "entries": entries, "skipped": skipped}
    with open(args.out_dir + "/" + PLAN_FILE, "w") as f:
        json.dump(plan, f, indent=1)
    print("done:", json.dumps(counts))


if __name__ == "__main__":
    main()