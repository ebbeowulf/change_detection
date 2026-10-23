import os
import re
import pickle
import numpy as np
from PIL import Image
import argparse
import pdb

def visualize_cv2(img, boxes, window_name="debug"):
    import cv2
    """
    img: numpy array (H,W,C) in uint8 or float32, or a PIL image
    boxes: Nx4 array-like [x_min, y_min, x_max, y_max] in pixel coords
    """

    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))[:, :, ::-1]
    else:
        img = np.asarray(img)

    img=np.ascontiguousarray(img)

    # Normalize float images to uint8
    if img.dtype != np.uint8:
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 255).astype(np.uint8)

    # Draw boxes
    for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[1])
            cv2.rectangle(img,(x1, y1),(x2, y2),color=(0, 0, 255),  thickness=2)

    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def build_segmenter(name, prompts, threshold):
    name = name.lower()

    if name == "sam3":
        from segmentation_utils.sam3_segmentation import sam3_segmentation
        return sam3_segmentation(prompts, threshold)

    elif name == "clip":
        from segmentation_utils.clip_segmentation import clip_segmentation
        return clip_segmentation(prompts, threshold)

    elif name == "dino":
        from segmentation_utils.dino_segmentation import dino_segmentation
        return dino_segmentation(prompts, threshold)

    else:
        raise ValueError(f"Unknown segmenter type: {name}")
    
def process_directory(input_dir, segmenter_name, prompts, threshold=0.2, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(input_dir, "segmentation_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Match color_<ID>.png
    pattern = re.compile(r"color_(\d+)\.png$", re.IGNORECASE)

    # Instantiate the chosen segmentation backend
    seg = build_segmenter(segmenter_name, prompts, threshold)

    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if not m:
            continue

        img_id = m.group(1)
        full_path = os.path.join(input_dir, fname)

        print(f"Processing {fname} (ID={img_id})")

        image = Image.open(full_path)

        outputs = seg.process_image(image, threshold=threshold)
        if outputs is None:
            print("  -> FAILED")
            continue

        # --- Save masks ---
        masks = {}
        for cls_idx, cls_name in enumerate(prompts):
            mask = seg.get_mask(cls_idx).cpu().numpy().astype(np.uint8)
            masks[cls_name] = mask

        mask_path = os.path.join(output_dir, f"mask_{img_id}.npy")
        np.save(mask_path, masks)
        print(f"  -> Saved masks to {mask_path}")

        # --- Save bounding boxes ---
        boxes = {}
        for cls_idx, cls_name in enumerate(prompts):
            boxes[cls_name] = seg.get_boxes(cls_idx)
            # visualize_cv2(image, boxes[cls_name], window_name=cls_name)

        box_path = os.path.join(output_dir, f"boxes_{img_id}.pkl")
        with open(box_path, "wb") as fh:
            pickle.dump(boxes, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  -> Saved boxes to {box_path}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing color_<ID>.png images")
    parser.add_argument("output_dir", type=str, help="Export directory to store segmentation masks")
    parser.add_argument("--backend", type=str, required=True,
                        choices=["sam3", "clip", "dino"],
                        help="Which segmentation backend to use")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="List of text prompts to search for")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    process_directory(
        args.input_dir,
        segmenter_name=args.backend,
        prompts=args.prompts,
        threshold=args.threshold,
        output_dir=args.output_dir
    )
