"""Quality assessment for the segment-by-exclusion pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class QualityReport:
    ok: bool
    coverage: float           # keep-mask coverage
    reasons: List[str]
    suggestion: str = ""


def _man_made_like(exclude_mask: np.ndarray) -> int:
    """Count elongated small blobs in the exclude mask (wires / poles / antennas).

    A positive count is actually a good sign — it means SAM 3 caught them —
    so this is diagnostic, not a failure condition. We only care when none
    are detected but the keep-mask still shows them (see ``_keep_has_wires``).
    """
    num, _, stats, _ = cv2.connectedComponentsWithStats(exclude_mask.astype(np.uint8), connectivity=8)
    count = 0
    H, W = exclude_mask.shape
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 64:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > 5 and area < (H * W) * 0.01:
            count += 1
    return count


def _top_edge_hits_frame_top(keep_mask: np.ndarray) -> float:
    """Fraction of columns whose topmost True pixel sits at row 0.

    If this is large, the sky wasn't excluded and the "ridge silhouette"
    isn't there.
    """
    H, W = keep_mask.shape
    col_any = keep_mask.any(axis=0)
    if not col_any.any():
        return 0.0
    # topmost True row per column; columns with no True get H
    first_true = np.argmax(keep_mask, axis=0)
    first_true = np.where(col_any, first_true, H)
    return float((first_true == 0).sum()) / float(W)


def _sky_color_leakage(keep_mask: np.ndarray, image_rgb: np.ndarray) -> bool:
    """Check if pixels in the top 5% rows inside keep_mask look like sky."""
    H = keep_mask.shape[0]
    band = max(1, int(0.05 * H))
    top_keep = keep_mask[:band]
    if top_keep.sum() < band * keep_mask.shape[1] * 0.1:
        return False  # not enough top-band kept to judge
    hsv = cv2.cvtColor(image_rgb[:band], cv2.COLOR_RGB2HSV)
    hs = hsv[top_keep]
    if hs.size == 0:
        return False
    mean_h = float(np.mean(hs[:, 0]))
    mean_s = float(np.mean(hs[:, 1]))
    mean_v = float(np.mean(hs[:, 2]))
    # Blue-sky gamut: hue ~95–130 (OpenCV 0–180), low-ish saturation, high value.
    return 95 <= mean_h <= 130 and mean_s < 120 and mean_v > 150


def assess(
    keep_mask: np.ndarray,
    exclude_mask: Optional[np.ndarray],
    image_rgb: np.ndarray,
    min_keep: float,
    max_keep: float,
) -> QualityReport:
    """Assess the keep mask; use the exclude mask for diagnostic hints."""
    reasons: List[str] = []
    H, W = keep_mask.shape
    coverage = float(keep_mask.sum()) / float(H * W)

    if coverage < min_keep:
        reasons.append(f"keep coverage {coverage:.4f} below min {min_keep}")
    if coverage > max_keep:
        reasons.append(f"keep coverage {coverage:.4f} exceeds max {max_keep}")

    # Sky/top-edge checks only fire when the kept region is already suspicious
    # (coverage > 0.85). Aerial straight-down shots legitimately keep the
    # whole top row when the ice extends to frame edge.
    if coverage > 0.85:
        top_hit = _top_edge_hits_frame_top(keep_mask)
        if top_hit > 0.95:
            reasons.append(
                f"top of frame kept across {top_hit:.0%} of columns — "
                "sky/background likely not excluded"
            )
        if _sky_color_leakage(keep_mask, image_rgb):
            reasons.append("sky-colored pixels remain in the kept region")

    suggestion = ""
    if reasons:
        if coverage > max_keep:
            suggestion = "add 'sky', 'clouds', 'distant mountains' to exclude prompts"
        elif coverage < min_keep:
            suggestion = "exclude list may be over-aggressive — remove ambiguous prompts"
        else:
            suggestion = "refine exclude wording (e.g. 'snow on distant peaks')"

    if exclude_mask is not None:
        # diagnostic only — not a failure signal
        _ = _man_made_like(exclude_mask)

    return QualityReport(ok=not reasons, coverage=coverage, reasons=reasons, suggestion=suggestion)
