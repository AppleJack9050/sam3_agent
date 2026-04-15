import numpy as np
from PIL import Image

from sam3_agent.input_module import (
    iter_tiles,
    load_image,
    normalize_resolution,
    should_tile,
    stitch_masks,
)


def test_load_image_rgb(ice_image_path, ice_image):
    loaded = load_image(ice_image_path, color_space="RGB")
    assert loaded.shape == ice_image.shape
    np.testing.assert_array_equal(loaded, ice_image)


def test_load_image_bgr(ice_image_path):
    loaded = load_image(ice_image_path, color_space="BGR")
    rgb = load_image(ice_image_path, color_space="RGB")
    np.testing.assert_array_equal(loaded[..., ::-1], rgb)


def test_normalize_resolution_noop_when_small(ice_image):
    out, scale = normalize_resolution(ice_image, target_long_side=1024)
    assert scale == 1.0
    assert out.shape == ice_image.shape


def test_normalize_resolution_downscales_when_large():
    big = np.zeros((4096, 2048, 3), dtype=np.uint8)
    out, scale = normalize_resolution(big, target_long_side=2048)
    assert scale == 0.5
    assert max(out.shape[:2]) == 2048


def test_should_tile():
    small = np.zeros((1024, 1024, 3), dtype=np.uint8)
    big = np.zeros((3000, 3000, 3), dtype=np.uint8)
    assert not should_tile(small, threshold=2048)
    assert should_tile(big, threshold=2048)


def test_iter_tiles_covers_full_image_with_overlap():
    img = np.zeros((1200, 1500, 3), dtype=np.uint8)
    tiles = list(iter_tiles(img, tile_size=512, overlap=64))
    assert tiles, "should produce at least one tile"
    # Every tile is full-sized for this image (image >= tile_size on both axes)
    for t in tiles:
        assert t.w == 512 and t.h == 512
    # Coverage: OR all tile footprints should cover the full image
    cover = np.zeros(img.shape[:2], dtype=bool)
    for t in tiles:
        cover[t.y : t.y + t.h, t.x : t.x + t.w] = True
    assert cover.all()


def test_iter_tiles_edge_tiles_shifted_inward():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    tiles = list(iter_tiles(img, tile_size=512, overlap=64))
    # Max x+w and y+h must not exceed image bounds
    for t in tiles:
        assert t.x + t.w <= 600
        assert t.y + t.h <= 600


def test_stitch_masks_ors_overlaps():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    tiles = list(iter_tiles(img, tile_size=512, overlap=64))
    pieces = []
    # Each tile fills its top-left 10x10 with True
    for t in tiles:
        m = np.zeros((t.h, t.w), dtype=bool)
        m[:10, :10] = True
        pieces.append((t, m))
    full = stitch_masks(pieces, img.shape[:2])
    # Every tile's (y, x) anchor should be True in the stitched mask
    for t in tiles:
        assert full[t.y, t.x]
