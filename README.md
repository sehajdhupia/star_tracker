# star_tracker

A tiny, self-contained **star detection and centroiding** demo. It synthesizes a star field, detects the “stars” as blobs, estimates their centers with **sub-pixel** precision, and compares detections to ground truth.

This is a minimal slice of what ground test software for a **star tracker** does: ingest images → detect stars → measure positions → report accuracy.

---

## What’s a star tracker?

A **star tracker** is a camera + software that images the night sky, identifies stars, and infers the spacecraft’s **attitude** (orientation). To do that reliably, it needs very precise star **centroids**—often **sub-pixel** accuracy.

---

## What this project does

1. **Generate a synthetic star field**  
   Each star is modeled with a small optical blur (a **Point Spread Function**, or **PSF**), using a **2D Gaussian**:

   \[
   I(x,y) = A \exp\!\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right) + B
   \]

   - \(A\): peak brightness (amplitude)  
   - \((x_0, y_0)\): true sub-pixel position  
   - \(\sigma\): PSF width in pixels  
   - \(B\): background level

2. **Detect stars**  
   Pixels above a chosen **threshold** are marked “bright,” then grouped into blobs via **8-connected components**.

3. **Compute sub-pixel centroids (center of mass)**  
   For each blob, compute intensity-weighted center:

   \[
   x_c=\frac{\sum x_i I_i}{\sum I_i},\qquad y_c=\frac{\sum y_i I_i}{\sum I_i}
   \]

   This yields sub-pixel star positions that are crucial for high-accuracy attitude estimation.

4. **Score accuracy**  
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