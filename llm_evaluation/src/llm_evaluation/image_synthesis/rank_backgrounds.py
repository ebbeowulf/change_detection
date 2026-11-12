"""Rank stage: order candidate backgrounds for a query by how well their scene
caption matches a fixed description of where the query is found.

  1. Screen the captioned images for the query (same any-word rule as the plan stage).
  2. Embed every caption with a sentence-embedding model (cached next to the captions,
     computed once and reused for every query).
  3. Embed QUERY_TEMPLATE filled with the query.
  4. Sort the screened images by cosine similarity to it, best first.

Output in out_dir:
  background_ranking.json   settings, template, screening summary, and every screened
                            image with its caption and similarity score, best first.
plan_insertions.py tries backgrounds in this order when the file is present in its out_dir.
"""
import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from screen_negatives import check_out_dir, index_texts, load_dataset, related_names, screen_images

# ---------- settings ----------
EMBED_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model
QUERY_TEMPLATE = "A photo of a place where a {query} would typically be found."
CAPTION_FILE = "scene_captions.json"      # in caption_dir (from caption_scenes.py)
EMBED_FILE = "caption_embeddings.npy"     # cache, written to caption_dir
EMBED_INDEX = "caption_embeddings.json"   # image order and model of the cache
RANK_FILE = "background_ranking.json"     # written to out_dir

SETTINGS = dict(EMBED_MODEL=EMBED_MODEL, QUERY_TEMPLATE=QUERY_TEMPLATE)


def caption_embeddings(model, caption_dir, captions):
    """Embeddings for every caption, in sorted image order; cached in caption_dir."""
    paths = sorted(captions)
    emb_path, idx_path = caption_dir + "/" + EMBED_FILE, caption_dir + "/" + EMBED_INDEX
    if os.path.exists(emb_path) and os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        if idx["model"] == EMBED_MODEL and idx["images"] == paths:
            return paths, np.load(emb_path)
    print("embedding %d captions" % len(paths))
    emb = model.encode([captions[p] for p in paths], batch_size=256,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(emb_path, emb)
    with open(idx_path, "w") as f:
        json.dump({"model": EMBED_MODEL, "images": paths}, f)
    return paths, emb


def main():
    ap = argparse.ArgumentParser(description="Rank captioned backgrounds for a query by caption similarity.")
    ap.add_argument("--dataset", type=str, default="coco", choices=["coco", "openimages"], help="Which dataset's annotations to use for screening")
    ap.add_argument("--caption_dir", type=str, required=True, help="Folder holding scene_captions.json")
    ap.add_argument("--ann_dir", type=str, required=True, help="Annotations directory (COCO annotations/ or the Open Images folder)")
    ap.add_argument("--query", type=str, required=True, help="Query text, e.g. 'train'")
    ap.add_argument("--ignore_words", type=str, default="", help="Comma-separated query words NOT used for screening, e.g. 'red,small'")
    ap.add_argument("--out_dir", type=str, required=True, help="Category results folder; the plan stage reads the ranking from here")
    args = ap.parse_args()

    with open(args.caption_dir + "/" + CAPTION_FILE) as f:
        cap = json.load(f)
    captions, img_dir = cap["captions"], cap["img_dir"]
    check_out_dir(args.out_dir, img_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Screen captioned images for the query
    texts, _, _ = load_dataset(args.dataset, img_dir, args.ann_dir)
    kept, report = screen_images(sorted(captions), index_texts(texts), args.query, args.ignore_words.split(","),
                                 related_names(args.dataset, args.ann_dir, args.query))
    print("%s: %d of %d captioned images kept after screening" % (args.query, len(kept), len(captions)))

    # 2-4. Embed and rank
    model = SentenceTransformer(EMBED_MODEL)
    paths, emb = caption_embeddings(model, args.caption_dir, captions)
    row = {p: i for i, p in enumerate(paths)}
    query_text = QUERY_TEMPLATE.format(query=args.query)
    q = model.encode([query_text], normalize_embeddings=True)[0]
    scores = emb[[row[p] for p in kept]] @ q
    order = np.argsort(-scores)
    ranking = [{"image": kept[i], "caption": captions[kept[i]], "score": round(float(scores[i]), 4)} for i in order]

    with open(args.out_dir + "/" + RANK_FILE, "w") as f:
        json.dump({"query": args.query, "query_text": query_text, "settings": SETTINGS,
                   "caption_file": args.caption_dir + "/" + CAPTION_FILE,
                   "num_captioned": len(captions), "screen_report": report, "ranking": ranking}, f, indent=1)
    print("top 5:")
    for r in ranking[:5]:
        print("  %.3f  %s" % (r["score"], r["caption"][:100]))


if __name__ == "__main__":
    main()