"""Output: blackout + save to PNG (alpha) or JPEG (black bg) + raw mask."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixelwise blackout: keep only pixels inside mask."""
    m = mask.astype(bool)
    out = image.copy()
    out[~m] = 0
    return out


def save_result(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    fmt: str = "png",
    save_raw_mask: bool = True,
) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = {}
    fmt = fmt.lower()
    blacked = apply_mask(image_rgb, mask)  # pixels outside mask = (0,0,0)
    if fmt == "png":
        Image.fromarray(blacked, mode="RGB").save(out_path.with_suffix(".png"))
        paths["image"] = str(out_path.with_suffix(".png"))
    elif fmt in ("jpg", "jpeg"):
        Image.fromarray(blacked, mode="RGB").save(out_path.with_suffix(".jpg"), quality=95)
        paths["image"] = str(out_path.with_suffix(".jpg"))
    else:
        raise ValueError(f"Unknown output format: {fmt}")

    if save_raw_mask:
        mask_path = out_path.with_name(out_path.stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        paths["mask"] = str(mask_path)

    return paths
