import numpy as np

from sam3_agent.quality import assess


def _blank_image(h=256, w=256):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _sky_ground_image(h=256, w=256):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = (120, 170, 230)
    img[h // 2 :] = (55, 50, 48)
    return img


def test_coverage_below_min_fails():
    mask = np.zeros((256, 256), dtype=bool)
    mask[0, 0] = True
    r = assess(mask, None, _blank_image(), min_keep=0.01, max_keep=0.9)
    assert not r.ok
    assert any("below" in s for s in r.reasons)


def test_coverage_above_max_fails():
    mask = np.ones((256, 256), dtype=bool)
    r = assess(mask, None, _blank_image(), min_keep=0.0, max_keep=0.5)
    assert not r.ok
    assert any("exceeds" in s for s in r.reasons)


def test_in_range_ok():
    mask = np.zeros((256, 256), dtype=bool)
    mask[80:176, 80:176] = True
    r = assess(mask, None, _blank_image(), min_keep=0.01, max_keep=0.9)
    assert r.ok, r.reasons


def test_top_edge_hits_frame_top_flagged():
    # Keep mask covers the whole frame → triggers gated sky checks.
    mask = np.ones((256, 256), dtype=bool)
    r = assess(mask, None, _blank_image(), min_keep=0.0, max_keep=1.0)
    assert not r.ok
    assert any("top of frame kept" in s for s in r.reasons)


def test_sky_color_leakage_flagged():
    # Keep whole frame of a sky image → triggers sky-color leakage check.
    mask = np.ones((256, 256), dtype=bool)
    img = _sky_ground_image()
    r = assess(mask, None, img, min_keep=0.0, max_keep=1.0)
    assert not r.ok
    assert any("sky" in s.lower() or "top of frame" in s for s in r.reasons)
