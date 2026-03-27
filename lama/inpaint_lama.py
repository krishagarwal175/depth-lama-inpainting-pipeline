import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add lama to path
sys.path.insert(0, r"D:\lama")

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf
import torch

WARP_DIR   = r"D:\Depth-Anything-V2\warped_output"
OUT_DIR    = r"D:\Depth-Anything-V2\inpainted_output"
MODEL_PATH = r"D:\lama\models\best.ckpt"
CONFIG_PATH= r"D:\lama\config.yaml"

os.makedirs(OUT_DIR, exist_ok=True)

# Load model
print("Loading LaMa model...")
train_config = OmegaConf.load(CONFIG_PATH)
train_config.training_model.predict_only = True
model = load_checkpoint(train_config, MODEL_PATH, strict=False, map_location="cpu")
model.eval()
print("Model loaded.")

def inpaint(img_bgr, mask_gray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (mask_gray > 0).astype(np.float32)

    img_t  = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        batch = {"image": img_t, "mask": mask_t}
        result = model(batch)
        out = result["inpainted"][0].permute(1,2,0).numpy()

    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

warped_files = sorted(Path(WARP_DIR).glob("*_warped.png"))
for warped_path in warped_files:
    mask_path = Path(WARP_DIR) / warped_path.name.replace("_warped.png", "_mask.png")
    if not mask_path.exists():
        print(f"  No mask for {warped_path.name}, skipping.")
        continue

    img  = cv2.imread(str(warped_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Check if image and mask sizes match
    img = cv2.imread(warped_path)
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    h, w = img.shape[:2]

    new_h = h - (h % 8)
    new_w = w - (w % 8)

    img = img[:new_h, :new_w]
    mask = mask[:new_h, :new_w]



    print(f"  Inpainting {warped_path.name}...")
    result = inpaint(img, mask)

    out_name = warped_path.name.replace("_warped.png", "_inpainted.png")
    cv2.imwrite(str(Path(OUT_DIR) / out_name), result)
    print(f"  Saved: {out_name}")

print("\nAll done. Check inpainted_output/")