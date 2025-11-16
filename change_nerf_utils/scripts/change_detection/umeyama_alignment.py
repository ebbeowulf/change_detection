import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import pdb
def load_colmap_camera_centers(images_txt_path):
    centers = {}
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):  # every other line
        line = lines[i].strip()
        elems = line.split()
        if len(elems) < 10:
            continue
        image_name = elems[-1]
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        center = -rot.T @ np.array([tx, ty, tz])
        centers[image_name] = center
    return centers

def load_reference_coordinates(camera_pose_txt_path):
    coords = {}
    with open(camera_pose_txt_path, 'r') as f:
        for line in f:
            elems = line.strip().split()
            if len(elems) != 4:
                continue
            try:
                xyz = list(map(float, elems[1:4]))
                coords[elems[0]] = np.array(xyz)
            except ValueError:
                continue  # skip header or malformed lines
    return coords

def umeyama_alignment(A, B):
    #https://medium.com/@oriolm25/how-to-align-3d-point-clouds-with-icp-538e12bf923f
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    pdb.set_trace()
    return c, R, t

# def umeyama_alignment(src, dst):
#     src_mean = src.mean(axis=0)
#     dst_mean = dst.mean(axis=0)
#     src_demean = src - src_mean
#     dst_demean = dst - dst_mean
#     cov = dst_demean.T @ src_demean / len(src)
#     U, S, Vt = np.linalg.svd(cov)
#     R_est = U @ Vt
#     if np.linalg.det(R_est) < 0:
#         U[:, -1] *= -1
#         R_est = U @ Vt
#     scale = np.trace(np.diag(S)) / np.sum(src_demean ** 2)
#     pdb.set_trace()
#     t = dst_mean - scale * R_est @ src_mean
#     return scale, R_est, t

def apply_transform(centers, scale, R_est, t):
    aligned = {}
    for name, c in centers.items():
        aligned[name] = scale * R_est @ c + t
    return aligned

def main(images_txt, camera_pose_txt, output_txt):
    colmap_centers = load_colmap_camera_centers(images_txt)
    ref_coords = load_reference_coordinates(camera_pose_txt)

    matched_names = sorted(set(colmap_centers) & set(ref_coords))
    if len(matched_names) < 3:
        raise ValueError("Need at least 3 matching reference points for alignment.")

    src = np.array([colmap_centers[name] for name in matched_names])
    dst = np.array([ref_coords[name] for name in matched_names])

    scale, R_est, t = umeyama_alignment(dst, src)
    print(f"Estimated scale: {scale:.6f}")
    print(f"Estimated rotation:\n{R_est}")
    print(f"Estimated translation: {t}")

    aligned_centers = apply_transform(colmap_centers, scale, R_est, t)

    with open(output_txt, 'w') as f:
        for name, xyz in aligned_centers.items():
            f.write(f"{name} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align COLMAP camera centers using Umeyama transform.")
    parser.add_argument("images_txt", help="Path to COLMAP images.txt (TXT format)")
    parser.add_argument("camera_pose_txt", help="Path to camera_pose.txt with real-world coordinates")
    parser.add_argument("output_txt", help="Path to save aligned camera centers")
    args = parser.parse_args()
    main(args.images_txt, args.camera_pose_txt, args.output_txt)