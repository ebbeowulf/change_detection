"""
Caption stage: write an exhaustive object inventory for a random subset of images.

The subset is a seeded random shuffle of every image under img_dir, truncated to
num_images. Because the shuffle is fixed, a later run with a larger num_images
keeps the same first images and only captions the new ones.

Each caption provides a short scene description followed by an exhaustive inventory 
of every visible object, including small, partly hidden, and background items.
These captions are later used to choose suitable backgrounds for each query.

Output in out_dir:
  object_captions.json    settings, prompt, and {image_path: caption}
Resumable: images already captioned are skipped.
"""
import argparse
import json
import os
import random
import time

import ollama

from screen_negatives import check_out_dir, find_images

# ---------- settings ----------
MODEL = "qwen3.5:35b-a3b-q4_K_M"      # Ollama model tag
TEMPERATURE = 0.0          # deterministic captions
SEED = 0                   # fixes the random subset
SAVE_EVERY = 50            # write the output file after this many new captions
OUT_FILE = "object_captions.json"

CAPTION_PROMPT = (
    "Provide a short description of the overall scene in this photograph, followed by an "
    "exhaustive inventory of every visible object. You must explicitly list all small, "
    "partly hidden, and background items. A comprehensive, itemized list of objects is "
    "preferred over a fluent caption."
)

SETTINGS = dict(MODEL=MODEL, TEMPERATURE=TEMPERATURE, SEED=SEED)


def caption(path):
    resp = ollama.chat(model=MODEL, think=False,
                       options={"temperature": TEMPERATURE, "seed": SEED},
                       messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [path]}])
    return resp["message"]["content"].strip()


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)   # never leaves a half-written file behind


def main():
    ap = argparse.ArgumentParser(description="Caption the objects of a random subset of images.")
    ap.add_argument("--img_dir", type=str, required=True, help="Image directory searched recursively, e.g. .../train2017")
    ap.add_argument("--out_dir", type=str, required=True, help="Results folder (must be outside img_dir)")
    ap.add_argument("--num_images", type=int, required=True, help="Size of the random subset to caption")
    args = ap.parse_args()

    check_out_dir(args.out_dir, args.img_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = args.out_dir + "/" + OUT_FILE

    paths = find_images(args.img_dir)
    random.Random(SEED).shuffle(paths)
    subset = paths[:args.num_images]

    data = {"img_dir": args.img_dir, "settings": SETTINGS, "prompt": CAPTION_PROMPT, "captions": {}}
    if os.path.exists(out_path):
        with open(out_path) as f:
            data["captions"] = json.load(f)["captions"]
    todo = [p for p in subset if p not in data["captions"]]
    print("%d images found, subset of %d, %d already captioned, %d to do"
          % (len(paths), len(subset), len(subset) - len(todo), len(todo)))

    t0 = time.time()
    for n, p in enumerate(todo, 1):
        try:
            data["captions"][p] = caption(p)
            print(data["captions"][p])
        except Exception as e:
            print("  error on %s: %s" % (p, str(e)[:200]))
            continue
        if n % SAVE_EVERY == 0:
            save(out_path, data)
            rate = (time.time() - t0) / n
            print("  %d / %d  (%.1fs per image, about %.1f h left)" % (n, len(todo), rate, rate * (len(todo) - n) / 3600))
    save(out_path, data)
    print("done: %d captions in %s" % (len(data["captions"]), out_path))


if __name__ == "__main__":
    main()