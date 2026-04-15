"""Input and preprocessing: normalization, color space, tiling for large imagery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class Tile:
    image: np.ndarray        # HxWx3 uint8 RGB
    x: int                   # left in parent
    y: int                   # top in parent
    w: int
    h: int


def load_image(path: str | Path, color_space: str = "RGB") -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if color_space.upper() == "BGR":
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def normalize_resolution(img: np.ndarray, target_long_side: int) -> Tuple[np.ndarray, float]:
    """Resize so long side == target_long_side. Returns (resized, scale)."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return img, 1.0
    scale = target_long_side / long_side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def should_tile(img: np.ndarray, threshold: int) -> bool:
    h, w = img.shape[:2]
    return max(h, w) > threshold


def iter_tiles(img: np.ndarray, tile_size: int, overlap: int) -> Iterator[Tile]:
    """Slice into overlapping tiles. Tiles on the edge are shifted inward to keep tile_size."""
    h, w = img.shape[:2]
    step = tile_size - overlap
    assert step > 0, "overlap must be smaller than tile_size"

    ys = list(range(0, max(1, h - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))

    seen = set()
    for y in ys:
        for x in xs:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            tw = min(tile_size, w - x)
            th = min(tile_size, h - y)
            yield Tile(image=img[y : y + th, x : x + tw].copy(), x=x, y=y, w=tw, h=th)


def stitch_masks(tiles: List[Tuple[Tile, np.ndarray]], full_shape: Tuple[int, int]) -> np.ndarray:
    """Merge per-tile boolean masks into a single mask via OR in overlap regions."""
    H, W = full_shape
    out = np.zeros((H, W), dtype=np.uint8)
    for tile, mask in tiles:
        if mask is None:
            continue
        m = mask.astype(np.uint8)
        out[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w] |= m[: tile.h, : tile.w]
    return out.astype(bool)
