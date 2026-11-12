"""Caption stage: write an exhaustive object inventory for a random subset of images.
Multithreaded to send concurrent requests to the Ollama server.
"""

import argparse
import json
import os
import random
import time
import threading
import concurrent.futures

import ollama

from screen_negatives import check_out_dir, find_images

# ---------- settings ----------
MODEL = "qwen3.5:35b"      # Ollama model tag
TEMPERATURE = 0.0          # deterministic captions
SEED = 0                   # fixes the random subset
SAVE_EVERY = 50            # write the output file after this many new captions
OUT_FILE = "object_captions.json"
MAX_WORKERS = 4            # Match this to your OLLAMA_NUM_PARALLEL setting

CAPTION_PROMPT = (
    "Provide a short description of the overall scene in this photograph, followed by an "
    "exhaustive inventory of every visible object. You must explicitly list all small, "
    "partly hidden, and background items. A comprehensive, itemized list of objects is "
    "preferred over a fluent caption."
)

SETTINGS = dict(MODEL=MODEL, TEMPERATURE=TEMPERATURE, SEED=SEED)
save_lock = threading.Lock()
print_lock = threading.Lock()


def caption(path):
    resp = ollama.chat(model=MODEL, think=False,
                       options={"temperature": TEMPERATURE, "seed": SEED},
                       messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [path]}])
    return resp["message"]["content"].strip()


def process_image(path):
    try:
        cap = caption(path)
        return path, cap, None
    except Exception as e:
        return path, None, e


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)   # never leaves a half-written file behind


def main():
    ap = argparse.ArgumentParser(description="Caption the objects of a random subset of images concurrently.")
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
    
    total_todo = len(todo)
    print(f"{len(paths)} images found, subset of {len(subset)}, {len(subset) - total_todo} already captioned, {total_todo} to do")

    t0 = time.time()
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(process_image, p): p for p in todo}
        
        # Process as they complete
        for future in concurrent.futures.as_completed(futures):
            p = futures[future]
            path, cap, err = future.result()
            
            with save_lock:
                done_count += 1
                if err:
                    with print_lock:
                        print(f"  error on {path}: {str(err)[:200]}")
                else:
                    data["captions"][path] = cap
                    with print_lock:
                        print(f"[{done_count}/{total_todo}] {path}: Success")
                    
                if done_count % SAVE_EVERY == 0 or done_count == total_todo:
                    save(out_path, data)
                    rate = (time.time() - t0) / done_count
                    with print_lock:
                        print(f"  --> Saved {done_count} / {total_todo}  ({rate:.1f}s per image overall, ~{rate * (total_todo - done_count) / 3600:.1f} h left)")

    print(f"done: {len(data['captions'])} captions in {out_path}")

if __name__ == "__main__":
    main()