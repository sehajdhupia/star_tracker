# star_tracker

A tiny, self-contained **star detection and centroiding** demo. It synthesizes a star field, detects the “stars” as blobs, estimates their centers with **sub-pixel** precision, and compares detections to ground truth.

This is a minimal slice of what ground test software for a **star tracker** does: ingest images → detect stars → measure positions → report accuracy.

---

## What’s a star tracker?

A **star tracker** is a camera + software that images the night sky, identifies stars, and infers the spacecraft’s **attitude** (orientation). To do that reliably, it needs very precise star **centroids**—often **sub-pixel** accuracy.

---

What this project does
----------------------

1.  **Generate a synthetic star field**\
    Each star is modeled with a small optical blur (a **Point Spread Function**, or **PSF**), using a **2D Gaussian**:

    ```
    I(x,y) = A exp(-((x-x₀)² + (y-y₀)²)/(2σ²)) + B
    ```

    -   `A`: peak brightness (amplitude)
    -   `(x₀, y₀)`: true sub-pixel position
    -   `σ`: PSF width in pixels
    -   `B`: background level
2.  **Detect stars**\
    Pixels above a chosen **threshold** are marked "bright," then grouped into blobs via **8-connected components**.
3.  **Compute sub-pixel centroids (center of mass)**\
    For each blob, compute intensity-weighted center:

    ```
    x_c = (Σ xᵢ Iᵢ) / (Σ Iᵢ),    y_c = (Σ yᵢ Iᵢ) / (Σ Iᵢ)
    ```

    This yields sub-pixel star positions that are crucial for high-accuracy attitude estimation.
4.  **Score accuracy**\
    Greedy nearest-neighbor **matching** between detected centroids and known truth positions → report **precision**, **recall**, and **mean error (pixels)**.
---

## Quick start

### Run the simple demo

```bash
# optional: create a virtual env
python3 -m venv .venv && source .venv/bin/activate

# install deps
pip install numpy pandas matplotlib

# run with defaults; outputs go to ./out_simple
python3 star_tracker_demo_simple.py
```

You'll get:

-   `out_simple/starfield_noiseless.png` --- synthetic star field (no noise)
-   `out_simple/detections_overlay_noiseless.png` --- stars with centroid markers
-   `out_simple/truth_positions_noiseless.csv` --- ground-truth positions
-   `out_simple/detections_noiseless.csv` --- detected centroids + flux/area
-   `out_simple/matches_noiseless.csv` --- truth↔detection pairs + pixel error

How to read the outputs
-----------------------

### Images

-   `starfield_noiseless.png`: "truth" image (sum of Gaussian PSFs + background).
-   `detections_overlay_noiseless.png`: same image with detected centroids overlaid.

### CSVs

**`truth_positions_noiseless.csv`** (generated truth)

-   `x_true`, `y_true` --- sub-pixel true positions
-   `amplitude` --- PSF brightness

**`detections_noiseless.csv`** (what the detector found)

-   `label` --- blob id (from connected components)
-   `x_det`, `y_det` --- sub-pixel centroid
-   `flux` --- sum of pixel intensities in the blob
-   `area` --- number of pixels in the blob (helps filter hot pixels in noisy mode)

**`matches_noiseless.csv`** (scoring)

-   `truth_idx`, `det_idx` --- row indices into the above tables
-   `x_true`, `y_true`, `x_det`, `y_det`
-   `err_px` --- Euclidean pixel error for that star

### Metrics you can quote

-   **Precision** = matched / detections
-   **Recall** = matched / truth
-   **Mean pixel error** = average of `err_px` across matches