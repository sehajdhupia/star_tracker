#!/usr/bin/env python3
"""
Star Field + Centroiding Demo
This has noise implementation

--------------------------------
Generates a synthetic star field, adds sensor-like noise and hot pixels,
detects stars via threshold + connected components, computes sub-pixel
intensity-weighted centroids, and matches detections to ground truth.
Outputs images and CSVs.

Usage:
    python star_tracker_demo.py --width 640 --height 480 --n_stars 40 --sigma_px 1.2 --k 3.0 --outdir ./out

Author: You (prepared for Rocket Lab interview)
"""
import argparse
import os
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- Utility functions ----------------
def generate_star_field(
    width=640,
    height=480,
    n_stars=40,
    sigma_px=1.2,
    amp_range=(150, 255),
    bg_level=20,
    rng_seed=42
):
    """
    Generate a clean star field (float32 image) and ground-truth positions.
    Each star is modeled as a 2D Gaussian PSF added onto a constant background.
    """
    rng = np.random.default_rng(rng_seed)
    img = np.ones((height, width), dtype=np.float32) * bg_level

    # avoid edges so PSF fits
    margin = int(4 * sigma_px + 3)
    xs = rng.uniform(margin, width - margin, n_stars)
    ys = rng.uniform(margin, height - margin, n_stars)
    amps = rng.uniform(amp_range[0], amp_range[1], n_stars)

    yy, xx = np.mgrid[0:height, 0:width]
    for x0, y0, a in zip(xs, ys, amps):
        # 2D Gaussian PSF
        g = a * np.exp(-(((xx - x0) ** 2) + ((yy - y0) ** 2)) / (2 * sigma_px ** 2))
        img += g

    truth = pd.DataFrame({"x_true": xs, "y_true": ys, "amplitude": amps})
    return img, truth


def add_noise_and_artifacts(img, read_noise_std=8.0, hot_pixel_prob=1e-4, rng_seed=123):
    """
    Add Gaussian read noise and occasional hot pixels.
    Simulates typical sensor read noise and sporadic hot pixels.
    """
    rng = np.random.default_rng(rng_seed)
    noisy = img + rng.normal(0, read_noise_std, img.shape)

    # hot pixels
    mask_hot = rng.uniform(0, 1, img.shape) < hot_pixel_prob
    noisy = noisy.copy()
    if mask_hot.any():
        noisy[mask_hot] += rng.uniform(200, 400, size=mask_hot.sum())

    # clip to [0, 65535] range (simulate 16-bit sensor, but keep float)
    noisy = np.clip(noisy, 0, 65535).astype(np.float32)
    return noisy


def threshold_image(img, k=3.0):
    """
    Global threshold at mean + k*std. Returns binary mask and stats.
    """
    mu = float(img.mean())
    sd = float(img.std(ddof=0))
    thr = mu + k * sd
    binary = img > thr
    return binary, thr, mu, sd


def connected_components_label(binary):
    """
    Simple 8-connected component labeling using BFS (pure numpy/python).
    Returns labels (int32) and number of components.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    # neighbors (8-connectivity)
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    visited = np.zeros_like(binary, dtype=bool)

    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                current_label += 1
                # BFS
                q = deque()
                q.append((y, x))
                visited[y, x] = True
                labels[y, x] = current_label
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                labels[ny, nx] = current_label
                                q.append((ny, nx))
    return labels, current_label


def compute_centroids(img, labels, min_pixels=5):
    """
    Intensity-weighted centroids per labeled blob.
    Returns DataFrame with x_det, y_det, flux, area, label.
    """
    df = []
    n = labels.max()
    for label in range(1, n + 1):
        mask = labels == label
        area = int(mask.sum())
        if area < min_pixels:
            continue
        # pixel coordinates
        ys, xs = np.nonzero(mask)
        intensities = img[mask]
        flux = float(intensities.sum())
        # intensity-weighted centroid
        x_c = float((xs * intensities).sum() / flux)
        y_c = float((ys * intensities).sum() / flux)
        df.append({"label": label, "x_det": x_c, "y_det": y_c, "flux": flux, "area": area})
    return pd.DataFrame(df)


def match_detections_to_truth(detections, truth, max_dist_px=3.0):
    """
    Greedy nearest-neighbor matching from truth to detections.
    Adds columns x_err, y_err, err_px.
    """
    truth = truth.reset_index(drop=True)
    if detections.empty:
        truth["matched"] = False
        return detections, truth, pd.DataFrame()

    det_used = np.zeros(len(detections), dtype=bool)
    matches = []

    for i, t in truth.reset_index().iterrows():
        tx, ty = float(t["x_true"]), float(t["y_true"])
        dx = detections["x_det"].to_numpy()
        dy = detections["y_det"].to_numpy()
        d2 = (dx - tx) ** 2 + (dy - ty) ** 2
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))
        if (not det_used[j]) and dist <= max_dist_px:
            det_used[j] = True
            matches.append({
                "truth_idx": i,
                "det_idx": j,
                "x_true": tx,
                "y_true": ty,
                "x_det": float(detections.iloc[j]["x_det"]),
                "y_det": float(detections.iloc[j]["y_det"]),
                "err_px": dist
            })

    matches_df = pd.DataFrame(matches)
    return detections, truth, matches_df


def save_images_and_csvs(clean_img, noisy_img, detections, truth, matches, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Save images
    plt.figure(figsize=(6, 5))
    plt.imshow(clean_img, cmap="gray")
    plt.title("Clean Star Field (Synthetic)")
    plt.axis("off")
    clean_path = os.path.join(outdir, "starfield_clean.png")
    plt.savefig(clean_path, bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(noisy_img, cmap="gray")
    plt.title("Noisy Star Field")
    plt.axis("off")
    noisy_path = os.path.join(outdir, "starfield_noisy.png")
    plt.savefig(noisy_path, bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(noisy_img, cmap="gray")
    if not detections.empty:
        plt.scatter(detections["x_det"], detections["y_det"], s=25)
    plt.title("Detections Overlay (centroids)")
    plt.axis("off")
    overlay_path = os.path.join(outdir, "detections_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", dpi=150)
    plt.close()

    # Save CSVs
    truth_path = os.path.join(outdir, "truth_positions.csv")
    det_path = os.path.join(outdir, "detections.csv")
    matches_path = os.path.join(outdir, "matches.csv")
    truth.to_csv(truth_path, index=False)
    detections.to_csv(det_path, index=False)
    matches.to_csv(matches_path, index=False)

    return clean_path, noisy_path, overlay_path, truth_path, det_path, matches_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--n_stars", type=int, default=40)
    ap.add_argument("--sigma_px", type=float, default=1.2, help="PSF sigma in pixels")
    ap.add_argument("--k", type=float, default=3.0, help="Threshold at mu + k*std")
    ap.add_argument("--min_pixels", type=int, default=6, help="Min connected pixels for a valid star")
    ap.add_argument("--read_noise_std", type=float, default=8.0)
    ap.add_argument("--hot_pixel_prob", type=float, default=1e-4)
    ap.add_argument("--outdir", type=str, default="./out")
    args = ap.parse_args()

    # 1) Generate clean star field + truth
    clean_img, truth = generate_star_field(
        width=args.width,
        height=args.height,
        n_stars=args.n_stars,
        sigma_px=args.sigma_px
    )

    # 2) Add noise + artifacts
    noisy_img = add_noise_and_artifacts(clean_img,
                                        read_noise_std=args.read_noise_std,
                                        hot_pixel_prob=args.hot_pixel_prob)

    # 3) Threshold + connected components
    binary, thr, mu, sd = threshold_image(noisy_img, k=args.k)
    labels, ncomp = connected_components_label(binary)

    # 4) Intensity-weighted centroids
    detections = compute_centroids(noisy_img, labels, min_pixels=args.min_pixels)

    # 5) Match to ground truth and compute error
    detections, truth, matches = match_detections_to_truth(detections, truth, max_dist_px=3.0)

    # 6) Save outputs
    clean_path, noisy_path, overlay_path, truth_path, det_path, matches_path = \
        save_images_and_csvs(clean_img, noisy_img, detections, truth, matches, args.outdir)

    # 7) Print summary stats
    precision = len(matches) / max(len(detections), 1)
    recall = len(matches) / max(len(truth), 1)
    mae = float(matches["err_px"].mean()) if len(matches) > 0 else float("nan")

    print("Outputs:")
    print("  Clean image:", clean_path)
    print("  Noisy image:", noisy_path)
    print("  Overlay image:", overlay_path)
    print("  Truth CSV:", truth_path)
    print("  Detections CSV:", det_path)
    print("  Matches CSV:", matches_path)
    print("\nSummary:")
    print(f"  Total truth stars: {len(truth)}")
    print(f"  Total detections : {len(detections)}")
    print(f"  Matched          : {len(matches)}")
    print(f"  Precision        : {precision:.3f}")
    print(f"  Recall           : {recall:.3f}")
    print(f"  Mean error (px)  : {mae:.3f}")
    print("\nThresholding:")
    print(f"  mu={mu:.2f}, std={sd:.2f}, thr=mu+{args.k}*std -> {mu + args.k*sd:.2f}")


if __name__ == "__main__":
    main()
