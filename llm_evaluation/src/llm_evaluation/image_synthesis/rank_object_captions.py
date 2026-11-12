"""Compare an input query to an exhaustive object inventory using sentence embeddings.

This script embeds an input query and candidate object captions using a sentence-transformer
model, and ranks the images based on cosine similarity. It provides a callable function 
for other programs and can be run directly to print the top N matches.
"""
import argparse
import json
import math
import os

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- settings ----------
EMBED_MODEL = "all-MiniLM-L6-v2"          # sentence-transformers model
QUERY_TEMPLATE = "A photograph of a scene containing {query}."
CAPTION_FILE = "object_captions.json"     # from caption_scenes_parallel.py
EMBED_FILE = "object_embeddings.npy"      # cache, written to caption_dir
EMBED_INDEX = "object_embeddings.json"    # image order and model of the cache


def caption_embeddings(model, caption_dir, captions):
    #Embeddings for every caption, in sorted image order; cached in caption_dir
    paths = sorted(captions)
    emb_path = os.path.join(caption_dir, EMBED_FILE)
    idx_path = os.path.join(caption_dir, EMBED_INDEX)
    
    if os.path.exists(emb_path) and os.path.exists(idx_path):
        with open(idx_path) as f:
            idx = json.load(f)
        if idx["model"] == EMBED_MODEL and idx["images"] == paths:
            return paths, np.load(emb_path)
            
    print("embedding %d object captions" % len(paths))
    emb = model.encode([captions[p] for p in paths], batch_size=256,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(emb_path, emb)
    with open(idx_path, "w") as f:
        json.dump({"model": EMBED_MODEL, "images": paths}, f)
    return paths, emb


def get_top_matches(query, caption_dir, n=5):
    """
    Ranks images based on how well their object captions match the query.
    
    Args:
        query (str): The search query.
        caption_dir (str): Directory containing the object captions and where embeddings are cached.
        n (int): Number of top matches to return.
        
    Returns:
        list: Top N matches as a list of dicts with 'image', 'caption', and 'score'.
    """
    caption_path = os.path.join(caption_dir, CAPTION_FILE)
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"Captions file not found at {caption_path}")

    with open(caption_path) as f:
        cap = json.load(f)
    captions = cap["captions"]

    model = SentenceTransformer(EMBED_MODEL)
    paths, emb = caption_embeddings(model, caption_dir, captions)
    
    query_text = QUERY_TEMPLATE.format(query=query)
    q = model.encode([query_text], normalize_embeddings=True)[0]
    
    # Calculate cosine similarity between query and all captions
    scores = emb @ q
    order = np.argsort(-scores)
    
    ranking = [{"image": paths[i], "caption": captions[paths[i]], "score": round(float(scores[i]), 4)} for i in order]
    
    return ranking[:n]


def display_image_grid(matches):
    #Displays a list of matched images in a single grid window using matplotlib.
    if not matches:
        print("No images to display.")
        return
        
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("\\nError: 'matplotlib' and 'Pillow' are required to display images.")
        print("Please install them using: pip install matplotlib pillow")
        return

    n = len(matches)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    # Ensure axes is always a flattened array for easy iteration
    if n == 1:
        axes = [axes]
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
        
    for i, match in enumerate(matches):
        ax = axes[i]
        img_path = match['image']
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Image not found\\n{os.path.basename(img_path)}", 
                    ha='center', va='center', wrap=True)
            
        ax.set_title(f"Score: {match['score']:.3f}", fontsize=10)
        ax.axis('off')
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Rank object-captioned images for a query by caption similarity.")
    ap.add_argument("--caption_dir", type=str, required=True, help="Folder holding object_captions.json")
    ap.add_argument("--query", type=str, required=True, help="Query text, e.g. 'a wooden desk with a lamp'")
    ap.add_argument("--n", type=int, default=5, help="Number of top matches to return")
    ap.add_argument("--display", action="store_true", help="Display the top matches in a grid window")
    args = ap.parse_args()

    try:
        top_matches = get_top_matches(args.query, args.caption_dir, n=args.n)
        
        print(f"\\nTop {args.n} matches for query: '{args.query}'\\n" + "-"*50)
        for i, r in enumerate(top_matches, 1):
            clean_caption = r['caption'][:150].replace('\\n', ' ')
            print(f"{i}. [Score: {r['score']:.3f}] {r['image']}")
            print(f"   Caption: {clean_caption}...\\n")
            
        if args.display:
            print("Opening image grid display...")
            display_image_grid(top_matches)
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
