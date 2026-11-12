"""Screen negatives (COCO or Open Images) so that none of them mention the query.

An image is excluded if any query word (stopwords removed) appears as a whole
word in its labels (COCO: category labels and captions; Open Images: box labels
and human-verified image-level labels). Plurals are reduced with a simple rule
applied identically to query and annotation text. For Open Images, the words of
the query's parent and child classes (and parts) in the class hierarchy are also
screening words, since an image labeled only "Tool" may contain a hammer.

Run alone to see how many images each query word excludes; plan_insertions.py
uses the same functions.
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict

# ---------- settings ----------
INST_ANN = "instances_train2017.json"   # COCO instance annotations (in ann_dir)
CAP_ANN = "captions_train2017.json"     # COCO captions (in ann_dir)
OI_BOX_ANN = "validation-annotations-bbox.csv"             # Open Images boxes (in ann_dir)
OI_LABEL_ANN = "oidv7-val-annotations-human-imagelabels.csv" # Open Images image-level labels
OI_CLASSES = "oidv7-class-descriptions-boxable.csv"          # class code -> name
OI_HIERARCHY = "bbox_labels_600_hierarchy.json"              # class hierarchy
REPORT_FILE = "screen_report.json"      # written to out_dir
IMG_EXTS = (".jpg", ".jpeg", ".png")
STOPWORDS = {
    "a", "an", "the", "of", "with", "and", "or", "in", "on", "at", "to", "for",
    "from", "by", "near", "next", "into", "onto", "over", "under", "is", "are",
    "it", "its", "this", "that", "some", "any", "one", "two", "being", "has",
    "have", "his", "her", "their",
}


def normalize_word(w):
    """Lowercase and strip simple plurals."""
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us")):
        return w[:-1]
    return w


def tokenize(text):
    """Whole words only, so 'car' never matches 'cart'."""
    return {normalize_word(t) for t in re.findall(r"[a-z0-9]+", text.lower())}


def query_words(query, ignore_words=()):
    ignore = {normalize_word(w.strip()) for w in ignore_words if w.strip()}
    return {w for w in tokenize(query) if w not in STOPWORDS and w not in ignore}


def find_images(root):
    """Recursively list image files under root."""
    paths = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.normpath(os.path.join(d, f)))
    return sorted(paths)


def load_coco(img_dir, ann_dir):
    """Read the COCO instance and caption files once. Returns
    texts:      {image_path: [category names + captions]}
    boxes:      {image_path: [{"label": name, "box": [x1, y1, x2, y2]}]}  (crowd regions skipped)
    categories: {category name: category id}"""
    texts, boxes = defaultdict(list), defaultdict(list)
    with open(ann_dir + "/" + INST_ANN) as f:
        inst = json.load(f)
    files = {im["id"]: os.path.normpath(img_dir + "/" + im["file_name"]) for im in inst["images"]}
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    for p in files.values():
        texts[p]  # every listed image gets a record, even with no objects
    for a in inst["annotations"]:
        p = files[a["image_id"]]
        texts[p].append(cats[a["category_id"]])
        if not a.get("iscrowd", 0):
            x, y, w, h = a["bbox"]
            boxes[p].append({"label": cats[a["category_id"]], "box": [x, y, x + w, y + h]})
    with open(ann_dir + "/" + CAP_ANN) as f:
        caps = json.load(f)
    files = {im["id"]: os.path.normpath(img_dir + "/" + im["file_name"]) for im in caps["images"]}
    for a in caps["annotations"]:
        texts[files[a["image_id"]]].append(a["caption"])
    return dict(texts), dict(boxes), {name: cid for cid, name in cats.items()}


def load_oi_classes(ann_dir):
    """Open Images {class code: name}, in file order."""
    names = {}
    with open(ann_dir + "/" + OI_CLASSES) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] != "LabelName":
                names[row[0]] = row[1]
    return names


def load_openimages(img_dir, ann_dir):
    """Read the Open Images box and image-level label files. Returns, like load_coco:
    texts:      {image_path: [class names]}  (box labels + human-verified positive labels)
    boxes:      {image_path: [{"label": name, "box": [x1, y1, x2, y2], "norm": True}]}
                in fractions of image width/height (group boxes skipped)
    categories: {class name: id}, id = 1-based position in the class description file"""
    names = load_oi_classes(ann_dir)
    texts, boxes = defaultdict(list), defaultdict(list)
    path = lambda image_id: os.path.normpath(img_dir + "/" + image_id + ".jpg")
    with open(ann_dir + "/" + OI_LABEL_ANN) as f:
        for row in csv.DictReader(f):
            if row["Confidence"] == "1" and row["LabelName"] in names:
                texts[path(row["ImageID"])].append(names[row["LabelName"]])
    with open(ann_dir + "/" + OI_BOX_ANN) as f:
        for row in csv.DictReader(f):
            if row["LabelName"] not in names:
                continue
            p, name = path(row["ImageID"]), names[row["LabelName"]]
            texts[p].append(name)
            if row.get("IsGroupOf", "0") != "1":
                boxes[p].append({"label": name, "norm": True,
                                 "box": [float(row["XMin"]), float(row["YMin"]),
                                         float(row["XMax"]), float(row["YMax"])]})
    return dict(texts), dict(boxes), {n: i + 1 for i, n in enumerate(names.values())}


def oi_related_names(ann_dir, query):
    """Names of the query's ancestors, descendants and parts (both directions) in the
    Open Images hierarchy; empty if the query is not an Open Images class name."""
    names = load_oi_classes(ann_dir)
    code = {n.lower(): c for c, n in names.items()}.get(query.lower())
    if code is None:
        return []
    with open(ann_dir + "/" + OI_HIERARCHY) as f:
        root = json.load(f)
    found = []

    def walk(node, ancestors):
        here = node["LabelName"]
        children = node.get("Subcategory", []) + node.get("Part", [])
        if here == code:
            found.extend(ancestors)
            stack = list(children)
            while stack:
                c = stack.pop()
                found.append(c["LabelName"])
                stack.extend(c.get("Subcategory", []) + c.get("Part", []))
        for c in children:
            walk(c, ancestors + [here])

    walk(root, [])
    return sorted({names[c] for c in found if c in names and c != code})


def load_dataset(dataset, img_dir, ann_dir):
    """(texts, boxes, categories) from the chosen dataset's loader."""
    if dataset == "coco":
        return load_coco(img_dir, ann_dir)
    if dataset == "openimages":
        return load_openimages(img_dir, ann_dir)
    raise ValueError("unknown dataset: " + dataset)


def related_names(dataset, ann_dir, query):
    """Extra class names whose words also screen images (Open Images hierarchy only)."""
    return oi_related_names(ann_dir, query) if dataset == "openimages" else []


def index_texts(texts):
    """{path: [texts]} -> {path: set of words}."""
    return {p: set().union(*[tokenize(t) for t in ts]) if ts else set() for p, ts in texts.items()}


def screen_images(paths, index, query, ignore_words=(), extra_names=()):
    """Return (kept_paths, report). Images with no annotation record are excluded.
    extra_names (e.g. related classes) add their words to the screening words."""
    words = query_words(query, ignore_words)
    for n in extra_names:
        words |= query_words(n, ignore_words)
    if not words:
        raise ValueError("no screening words left in query: " + query)
    kept, per_word = [], Counter()
    excluded, no_record = 0, 0
    for p in paths:
        if p not in index:
            no_record += 1
            continue
        hits = words & index[p]
        if hits:
            excluded += 1
            per_word.update(hits)
        else:
            kept.append(p)
    report = {
        "query": query, "screen_words": sorted(words), "ignore_words": [w for w in ignore_words if w],
        "related_names": list(extra_names),
        "images_found": len(paths), "kept": len(kept), "excluded_by_text": excluded,
        "excluded_no_record": no_record, "excluded_per_word": dict(per_word.most_common()),
    }
    return kept, report


def check_out_dir(out_dir, data_dir):
    """Refuse to write results inside the source data directory."""
    o, d = os.path.realpath(out_dir), os.path.realpath(data_dir)
    if o == d or o.startswith(d + os.sep):
        raise SystemExit("output folder must be outside the data directory: " + data_dir)


def main():
    ap = argparse.ArgumentParser(description="Screen negatives for a query and write a report.")
    ap.add_argument("--dataset", type=str, default="coco", choices=["coco", "openimages"], help="Which dataset's annotations to use")
    ap.add_argument("--neg_dir", type=str, required=True, help="Image directory, e.g. .../train2017 or .../openimages/validation")
    ap.add_argument("--ann_dir", type=str, required=True, help="Annotations directory (COCO annotations/ or the Open Images folder)")
    ap.add_argument("--query", type=str, required=True, help="Query text, e.g. 'forklift'")
    ap.add_argument("--ignore_words", type=str, default="", help="Comma-separated query words NOT used for screening, e.g. 'red,small'")
    ap.add_argument("--out_dir", type=str, required=True, help="Results folder (must be outside neg_dir)")
    args = ap.parse_args()

    check_out_dir(args.out_dir, args.neg_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    texts, _, _ = load_dataset(args.dataset, args.neg_dir, args.ann_dir)
    _, report = screen_images(find_images(args.neg_dir), index_texts(texts), args.query, args.ignore_words.split(","),
                              related_names(args.dataset, args.ann_dir, args.query))
    with open(args.out_dir + "/" + REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()