import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Blur Metrics ---
def laplacian_variance(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def tenengrad(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    return np.mean(gx**2 + gy**2)

def fft_high_freq_energy(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    center = (h // 2, w // 2)
    radius = min(h, w) // 10
    # mask = np.ones_like(magnitude)
    mask = np.ascontiguousarray(np.ones_like(magnitude, dtype=np.uint8))
    cv2.circle(mask, center, radius, 0, -1)
    high_freq_energy = np.sum(magnitude * mask)
    return high_freq_energy / (h * w)

# --- Load and Score Images ---
def compute_blur_scores(image_dir,keyword):
    scores = []
    cnt=0
    for fname in os.listdir(image_dir):
        # If keyword is present - search for common image types
        if keyword in fname:
            print(fname)
            path = os.path.join(image_dir, fname)
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                scores.append({
                    'filename': fname,
                    'laplacian': laplacian_variance(img),
                    'tenengrad': tenengrad(img),
                    'fft_energy': fft_high_freq_energy(img)
                })
            except Exception as e:
                continue
    return scores

# --- Plotting ---
def plot_scores(scores):
    filenames = [s['filename'] for s in scores]
    lap = [s['laplacian'] for s in scores]
    ten = [s['tenengrad'] for s in scores]
    fft = [s['fft_energy'] for s in scores]

    x = np.arange(len(filenames))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, lap, width, label='Laplacian')
    plt.bar(x, ten, width, label='Tenengrad')
    plt.bar(x + width, fft, width, label='FFT Energy')
    plt.xticks(x, filenames, rotation=45, ha='right')
    plt.ylabel('Blur Score')
    plt.title('Blur Scores by Metric')
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np

def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def get_sharp_images(scores, keep_fraction=0.7):
    filenames = [s['filename'] for s in scores]
    lap = [s['laplacian'] for s in scores]
    ten = [s['tenengrad'] for s in scores]
    fft = [s['fft_energy'] for s in scores]

    lap_norm = normalize(lap)
    ten_norm = normalize(ten)
    fft_norm = normalize(fft)

    blur_score = (lap_norm + ten_norm + fft_norm) / 3
    sorted_indices = np.argsort(blur_score)
    cutoff = int(len(blur_score) * (1 - keep_fraction))
    keep_indices = sorted_indices[cutoff:]

    return [filenames[i] for i in keep_indices]

# --- Copy Files ---
def copy_images(filenames, src_dir, dst_dir):
    import shutil

    os.makedirs(dst_dir, exist_ok=True)
    for fname in filenames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copy2(src, dst)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter out blurry images using multiple metrics.")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save sharp images")
    parser.add_argument("--keep_fraction", type=float, default=0.7, help="Fraction of sharpest images to keep (default: 0.7)")
    parser.add_argument("--keyword",type=str,default="png",help="Keyword to use in retrieving images (default=png)")
    args = parser.parse_args()

    scores = compute_blur_scores(args.input_dir,args.keyword)
    # plot_scores(scores)
    sharp_images = get_sharp_images(scores, keep_fraction=args.keep_fraction)
    copy_images(sharp_images, args.input_dir, args.output_dir)

    print(f"Copied {len(sharp_images)} sharp images to {args.output_dir}")

if __name__ == "__main__":
    main()
