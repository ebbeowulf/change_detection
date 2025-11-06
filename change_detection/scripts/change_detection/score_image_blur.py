import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def laplacian_blur_score(image: np.ndarray) -> float:
    """
    Computes a blur score for an image using the variance of the Laplacian.
    Lower scores indicate more blur.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def fft_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    high_freq_energy = np.sum(magnitude_spectrum[gray.shape[0]//4:-gray.shape[0]//4,
                                                  gray.shape[1]//4:-gray.shape[1]//4])
    return high_freq_energy / gray.size

def tenengrad_blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(magnitude)

def blur_score(image: np.ndarray, method: int=2) -> float:
    if method==0:
        return laplacian_blur_score(image)
    elif method==1:
        return fft_blur_score(image)
    elif method==2:
        return tenengrad_blur_score(image)
    return None

def score_images_in_directory(directory: str) -> dict:
    """
    Scores all .jpg images in the given directory using the blur_score function.

    Returns:
        dict: Mapping of filename to blur score.
    """
    scores = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
            path = os.path.join(directory, filename)
            image = cv2.imread(path)
            if image is not None:
                score = blur_score(image)
                scores[filename] = score
            else:
                print(f"Warning: Could not read {filename}")
    return scores

def plot_histogram(scores: dict):
    """
    Plots a histogram of blur scores.
    """
    values = list(scores.values())
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Image Blur Scores")
    plt.xlabel("Blur Score (Variance of Laplacian)")
    plt.ylabel("Number of Images")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score and visualize image blurriness.")
    parser.add_argument("directory", help="Path to directory containing .jpg images")
    args = parser.parse_args()

    scores = score_images_in_directory(args.directory)
    for fname, score in sorted(scores.items(), key=lambda x: x[1]):
        print(f"{fname}: {score:.2f}")
    plot_histogram(scores)
