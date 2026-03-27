# Depth-Guided Image Inpainting Pipeline (Depth Anything V2 + LaMa)

## Overview

This project implements a depth-aware image inpainting pipeline by combining **Depth Anything V2** for monocular depth estimation with **LaMa** (Large Mask Inpainting) for high-quality image completion.

The pipeline generates novel views via depth-based warping and fills missing regions using a state-of-the-art inpainting model.

---

## Pipeline Architecture

Input Image → Depth Estimation → Warping → Mask Generation → LaMa Inpainting → Final Output

---

## Components

### 1. Depth Estimation (Depth Anything V2)

* Script: `Depth-Anything-V2/run.py`
* Generates dense depth maps from RGB images

### 2. Warping & Mask Generation

* Script: `Depth-Anything-V2/warp.py`
* Uses depth maps to create warped views and corresponding hole masks

### 3. Inpainting (LaMa)

* Script: `lama/inpaint_lama.py`
* Fills missing regions using learned contextual priors

---

## Directory Structure

```
depth-lama-inpainting-pipeline/

Depth-Anything-V2/        # Depth estimation + warping
lama/                     # Inpainting module

data/
  input_images/           # Original images
  depth_maps/             # Generated depth maps
  warped/                 # Warped images + masks
  inpainted/              # Final outputs

results/                  # Optional visualization outputs
```

---

## Setup Instructions

### 1. Clone repository

```
git clone https://github.com/krishagarwal175/depth-lama-inpainting-pipeline.git
cd depth-lama-inpainting-pipeline
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Download pretrained models

#### Depth Anything V2

Download model weights and place inside:

```
Depth-Anything-V2/checkpoints/
```

#### LaMa

Download LaMa checkpoint and place inside:

```
lama/models/
```

---

## Usage

### Step 1: Generate depth maps

```
python Depth-Anything-V2/run.py --encoder vits --img-path data/input_images --outdir data/depth_maps --pred-only
```

---

### Step 2: Generate warped images and masks

```
python Depth-Anything-V2/warp.py
```

---

### Step 3: Run LaMa inpainting

```
python lama/inpaint_lama.py
```

---

## Output

Final inpainted images are stored in:

```
data/inpainted/
```

---

## Key Features

* Depth-aware scene understanding
* Novel view synthesis via warping
* High-quality semantic inpainting
* Modular pipeline design

---

## Limitations

* Requires correctly aligned image, depth, and mask resolutions
* Sensitive to depth estimation quality
* No real-time processing

---

## Future Improvements

* Real-time inference optimization
* GUI for interactive usage
* Multi-view consistency
* Integration with NeRF / 3DGS pipelines

---

## Author

Krish Agarwal
GitHub: https://github.com/krishagarwal175
