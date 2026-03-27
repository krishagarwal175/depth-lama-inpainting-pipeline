import cv2
import numpy as np
import os
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
IMG_DIR    = "my_images"
DEPTH_DIR  = "depth_output"
OUT_DIR    = "warped_output"
os.makedirs(OUT_DIR, exist_ok=True)

FX, FY = 800.0, 800.0
CX, CY = 475.0, 630.0

OFFSETS = [
    ("left",  np.array([-0.05, 0.0, 0.0])),
    ("right", np.array([ 0.05, 0.0, 0.0])),
    ("up",    np.array([ 0.0, -0.03, 0.0])),
]
# ─────────────────────────────────────────────────────────────────────────────

def warp(src_img, depth_gray, t_rel):
    h, w = src_img.shape[:2]
    depth = depth_gray.astype(np.float32) / 255.0 * 5.0

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (u - CX) * Z / FX
    Y = (v - CY) * Z / FY
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    pts_new = pts + t_rel

    Z_new = pts_new[:, 2].clip(min=1e-6)
    u_new = (pts_new[:, 0] * FX / Z_new + CX).astype(int)
    v_new = (pts_new[:, 1] * FY / Z_new + CY).astype(int)

    warped = np.zeros_like(src_img)
    valid = (u_new >= 0) & (u_new < w) & (v_new >= 0) & (v_new < h)
    warped[v_new[valid], u_new[valid]] = src_img.reshape(-1, 3)[valid]

    kernel = np.ones((3, 3), np.uint8)
    warped_filled = cv2.dilate(warped, kernel, iterations=1)

    painted_mask = (warped.sum(axis=-1) > 0)
    warped_filled[painted_mask] = warped[painted_mask]

    hole_mask = (warped_filled.sum(axis=-1) == 0).astype(np.uint8) * 255
    hole_mask = cv2.dilate(hole_mask, kernel, iterations=2)

    return warped_filled, hole_mask


# ── MAIN PIPELINE ────────────────────────────────────────────────────────────

img_files = sorted(Path(IMG_DIR).glob("*.*"))

for img_path in img_files:
    depth_path = Path(DEPTH_DIR) / (img_path.stem + ".png")
    if not depth_path.exists():
        print(f"Depth not found for {img_path.name}, skipping.")
        continue

    src = cv2.imread(str(img_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)

    depth = cv2.resize(depth, (src.shape[1], src.shape[0]))

    for name, t in OFFSETS:

        warped, mask = warp(src, depth, t)

        out_stem = f"{img_path.stem}_{name}"

        warped_path = str(Path(OUT_DIR) / f"{out_stem}_warped.png")
        mask_path = str(Path(OUT_DIR) / f"{out_stem}_mask.png")

        cv2.imwrite(warped_path, warped)
        cv2.imwrite(mask_path, mask)


print("\nDone. Check warped_output/")