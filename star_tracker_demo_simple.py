#!/usr/bin/env python3
"""
Minimal Star Field + Centroiding
--------------------------------------------
Generates a star field using Gaussian PSFs, detects blobs with a fixed
absolute threshold, computes intensity-weighted centroids, and saves outputs.
"""
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

def generate_star_field(width=640, height=480, n_stars=40, sigma_px=1.2, amp_range=(150,255), bg_level=20, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    img = np.ones((height, width), dtype=np.float32) * bg_level
    margin = int(4 * sigma_px + 3)
    xs = rng.uniform(margin, width - margin, n_stars)
    ys = rng.uniform(margin, height - margin, n_stars)
    amps = rng.uniform(amp_range[0], amp_range[1], n_stars)
    yy, xx = np.mgrid[0:height, 0:width]
    for x0, y0, a in zip(xs, ys, amps):
        img += a * np.exp(-(((xx - x0) ** 2) + ((yy - y0) ** 2)) / (2 * sigma_px ** 2))
    truth = pd.DataFrame({"x_true": xs, "y_true": ys, "amplitude": amps})
    return img, truth, bg_level

def threshold_image_absolute(img, thr):
    return (img > thr)

def connected_components_label(binary):
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    visited = np.zeros_like(binary, dtype=bool)
    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                current_label += 1
                q = deque([(y,x)])
                visited[y, x] = True
                labels[y, x] = current_label
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            labels[ny, nx] = current_label
                            q.append((ny, nx))
    return labels

def compute_centroids(img, labels, min_pixels=5):
    out = []
    n = labels.max()
    for label in range(1, n+1):
        mask = labels == label
        area = int(mask.sum())
        if area < min_pixels:
            continue
        ys, xs = np.nonzero(mask)
        intensities = img[mask]
        flux = float(intensities.sum())
        x_c = float((xs * intensities).sum() / flux)
        y_c = float((ys * intensities).sum() / flux)
        out.append({"label": label, "x_det": x_c, "y_det": y_c, "flux": flux, "area": area})
    return pd.DataFrame(out)

def match_detections_to_truth(detections, truth, max_dist_px=3.0):
    if detections.empty:
        return pd.DataFrame()
    det_used = np.zeros(len(detections), dtype=bool)
    matches = []
    for i, t in truth.reset_index().iterrows():
        tx, ty = float(t["x_true"]), float(t["y_true"])
        dx = detections["x_det"].to_numpy()
        dy = detections["y_det"].to_numpy()
        d2 = (dx - tx)**2 + (dy - ty)**2
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))
        if (not det_used[j]) and dist <= max_dist_px:
            det_used[j] = True
            matches.append({
                "truth_idx": i, "det_idx": j,
                "x_true": tx, "y_true": ty,
                "x_det": float(detections.iloc[j]["x_det"]), "y_det": float(detections.iloc[j]["y_det"]),
                "err_px": dist
            })
    return pd.DataFrame(matches)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--n_stars", type=int, default=40)
    ap.add_argument("--sigma_px", type=float, default=1.2)
    ap.add_argument("--thr", type=float, default=70.0, help="Absolute threshold (default ~ bg 20 + 50)")
    ap.add_argument("--min_pixels", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="./out_simple")
    args = ap.parse_args()

    img, truth, bg = generate_star_field(args.width, args.height, args.n_stars, args.sigma_px)
    binary = threshold_image_absolute(img, args.thr)
    labels = connected_components_label(binary)
    detections = compute_centroids(img, labels, min_pixels=args.min_pixels)
    matches = match_detections_to_truth(detections, truth, max_dist_px=3.0)

    os.makedirs(args.outdir, exist_ok=True)
    plt.figure(figsize=(6,5)); plt.imshow(img, cmap="gray"); plt.title("Star Field"); plt.axis("off")
    plt.savefig(os.path.join(args.outdir, "starfield.png"), bbox_inches="tight", dpi=150); plt.close()
    plt.figure(figsize=(6,5)); plt.imshow(img, cmap="gray")
    if not detections.empty: plt.scatter(detections["x_det"], detections["y_det"], s=25)
    plt.title("Detections Overlay"); plt.axis("off")
    plt.savefig(os.path.join(args.outdir, "detections_overlay.png"), bbox_inches="tight", dpi=150); plt.close()

    truth.to_csv(os.path.join(args.outdir, "truth_positions.csv"), index=False)
    detections.to_csv(os.path.join(args.outdir, "detections.csv"), index=False)
    matches.to_csv(os.path.join(args.outdir, "matches.csv"), index=False)

    precision = len(matches) / max(len(detections), 1)
    recall = len(matches) / max(len(truth), 1)
    mae = float(matches["err_px"].mean()) if len(matches) else float("nan")
    print(f"thr={args.thr:.2f}, stars={len(truth)}, det={len(detections)}, matched={len(matches)}, "
          f"precision={precision:.3f}, recall={recall:.3f}, mean_err_px={mae:.3f}")

if __name__ == "__main__":
    main()
