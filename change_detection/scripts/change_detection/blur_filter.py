import cv2
import numpy as np
import os
import shutil
from score_image_blur import blur_score

def score_and_filter_images(src_dir: str, dst_dir: str, threshold: float, method: int) -> dict:
    """
    Scores all .jpg images in the source directory and copies those with blur score
    above the threshold to the destination directory.

    Returns:
        dict: Mapping of filename to blur score (only for copied images).
    """
    os.makedirs(dst_dir, exist_ok=True)
    filtered_scores = {}

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(".jpg"):
            src_path = os.path.join(src_dir, filename)
            image = cv2.imread(src_path)
            if image is not None:
                score = blur_score(image,method)
                if score >= threshold:
                    dst_path = os.path.join(dst_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    filtered_scores[filename] = score
            else:
                print(f"Warning: Could not read {filename}")
    return filtered_scores

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter sharp images by blur score.")
    parser.add_argument("src_dir", help="Path to source directory containing .jpg images")
    parser.add_argument("dst_dir", help="Path to destination directory for sharp images")
    parser.add_argument("--threshold", type=float, default=40.0,
                        help="Blur score threshold (default: 40.0)")
    parser.add_argument("--method", type=int, default=2,
                        help="Blur score method to use (laplacian=0, fft=1, tenengrad=2), default: 2")
    args = parser.parse_args()

    results = score_and_filter_images(args.src_dir, args.dst_dir, args.threshold, args.method)
    print(f"\nCopied {len(results)} images with blur score ≥ {args.threshold}:\n")
    for fname, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{fname}: {score:.2f}")
