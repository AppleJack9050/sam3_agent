from pathlib import Path

import numpy as np
from PIL import Image

from sam3_agent.output import apply_mask, save_result


def _img_and_mask():
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    return img, mask


def test_apply_mask_blacks_out_outside():
    img, mask = _img_and_mask()
    out = apply_mask(img, mask)
    assert (out[~mask] == 0).all()
    np.testing.assert_array_equal(out[mask], img[mask])


def test_save_png_writes_rgb_with_pure_black_outside_mask(tmp_path: Path):
    img, mask = _img_and_mask()
    paths = save_result(img, mask, tmp_path / "res", fmt="png", save_raw_mask=True)
    png = Image.open(paths["image"])
    assert png.mode == "RGB"
    arr = np.array(png)
    assert arr.shape == (64, 64, 3)
    # Pixels outside the mask must be pure black (0,0,0).
    assert (arr[~mask] == 0).all()
    # Pixels inside are preserved (PNG is lossless).
    np.testing.assert_array_equal(arr[mask], img[mask])
    assert Path(paths["mask"]).exists()


def test_save_jpeg_blacks_outside_mask(tmp_path: Path):
    img, mask = _img_and_mask()
    paths = save_result(img, mask, tmp_path / "res", fmt="jpeg", save_raw_mask=False)
    jpg = np.array(Image.open(paths["image"]).convert("RGB"))
    # JPEG is lossy — use tolerant check
    assert jpg[~mask].max() < 20
    assert "mask" not in paths
