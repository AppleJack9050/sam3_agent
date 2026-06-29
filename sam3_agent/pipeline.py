"""Deterministic orchestration for the SAM 3 segment-by-exclusion agent.

This replaces the old LangGraph ``StateGraph``. The control flow was always
deterministic (route on the quality gate, not on an LLM), so it is expressed
directly as a plain Python retry loop — no graph framework required:

    preprocess → build prompts → sam3_infer → quality ─ ok ──────────┐
                                     ▲             │                  ▼
                                     └─ adapt ◀─ fail (retry left) → save

The single genuinely LLM-driven step (``adapt``) goes through the Claude Agent
SDK in :mod:`sam3_agent.llm`, authenticated against your Claude subscription.

In `exclude` mode (default) the pipeline queries SAM 3 once per prompt in
`cfg.exclude_prompts`, unions the masks, inverts, morph-closes, and keeps the
largest connected component. In `include` mode it keeps SAM 3's positive mask
directly (legacy behavior).
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from . import llm
from .config import AgentConfig
from .input_module import (
    Tile,
    iter_tiles,
    load_image,
    normalize_resolution,
    should_tile,
    stitch_masks,
)
from .llm import PromptSetDecision
from .output import save_result
from .quality import QualityReport, assess
from .sam3_inference import Detection, SAM3Predictor


# ---------- Orchestration ----------

def run_pipeline(
    predictor: SAM3Predictor,
    image_path: str,
    output_path: str,
    cfg: AgentConfig,
) -> dict:
    """Run the full segment-by-exclusion pipeline and save the result.

    Returns a dict with: ``output_paths``, ``keep_mask``, ``exclude_mask``,
    ``quality`` (:class:`QualityReport`), ``attempt``, ``prompts``, ``history``.
    """
    # --- preprocess ---
    img = load_image(image_path, color_space=cfg.color_space)
    img, _ = normalize_resolution(img, cfg.target_long_side)

    # --- initial prompt set ---
    prompts = _initial_prompts(cfg)

    attempt = 0
    history: List[dict] = []
    keep_mask = exclude_mask = None
    report: QualityReport

    while True:
        # --- sam3 inference (+ optional color priors, invert, postprocess) ---
        keep_mask, exclude_mask = _segment(predictor, img, prompts, cfg)
        attempt += 1

        # --- quality gate ---
        report = assess(
            keep_mask,
            exclude_mask,
            img,
            cfg.min_keep_coverage,
            cfg.max_keep_coverage,
        )
        history.append({
            "attempt": attempt,
            "mode": cfg.mode,
            "prompts": list(prompts),
            "coverage": report.coverage,
            "ok": report.ok,
            "reasons": report.reasons,
        })

        # --- route: stop on success or when retries are exhausted ---
        if report.ok or attempt >= cfg.max_retries:
            break

        # --- adapt: Claude (subscription) or deterministic fallback ---
        decision = _decide_next(cfg, attempt, prompts, report.reasons)
        prompts = decision.prompts

    # --- save ---
    paths = save_result(
        img, keep_mask, output_path,
        fmt=cfg.output_format, save_raw_mask=cfg.save_raw_mask,
    )

    return {
        "output_paths": paths,
        "keep_mask": keep_mask,
        "exclude_mask": exclude_mask,
        "quality": report,
        "attempt": attempt,
        "prompts": prompts,
        "history": history,
    }


# ---------- steps ----------

def _initial_prompts(cfg: AgentConfig) -> List[str]:
    if cfg.mode == "include":
        if cfg.target:
            return [cfg.target]  # single-prompt override
        return list(cfg.include_prompts)
    return list(cfg.exclude_prompts)


def _segment(
    predictor: SAM3Predictor,
    img: np.ndarray,
    prompts: List[str],
    cfg: AgentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run SAM 3 over every prompt, union, then build keep/exclude masks."""
    raw = np.zeros(img.shape[:2], dtype=bool)
    for prompt in prompts:
        raw |= _infer_one(predictor, img, prompt, cfg)

    if cfg.mode == "exclude":
        # Augment SAM 3's output with deterministic color priors so the sky and
        # yellow placards are excluded even when SAM 3 misses them.
        if cfg.color_prior_sky:
            raw |= _color_prior_sky(img)
        if cfg.color_prior_yellow:
            raw |= _color_prior_yellow(img)
        keep = ~raw
        exclude = raw
    else:
        keep = raw
        exclude = np.zeros_like(raw)

    keep = _postprocess_keep(keep, cfg)
    return keep, exclude


def _decide_next(
    cfg: AgentConfig,
    attempt: int,
    last_prompts: List[str],
    reasons: List[str],
) -> PromptSetDecision:
    """Pick the next prompt set: Claude via the Agent SDK when ``use_llm`` is on
    (and reachable), otherwise cycle through the deterministic fallback sets.
    """
    fallbacks = list(
        cfg.include_prompt_fallbacks if cfg.mode == "include"
        else cfg.exclude_prompt_fallbacks
    )
    if cfg.use_llm:
        try:
            return llm.decide_next_prompts(cfg, attempt, last_prompts, reasons, fallbacks)
        except Exception:
            pass  # SDK/CLI/auth unavailable → deterministic fallback below

    idx = min(attempt - 1, len(fallbacks) - 1) if fallbacks else 0
    prompts = list(fallbacks[idx]) if fallbacks else last_prompts
    return PromptSetDecision(prompts=prompts, reasoning="deterministic fallback")


# ---------- helpers ----------

def _infer_one(predictor: SAM3Predictor, img: np.ndarray, prompt: str, cfg: AgentConfig) -> np.ndarray:
    if should_tile(img, cfg.tile_threshold):
        tiles_out: List[Tuple[Tile, np.ndarray]] = []
        for tile in iter_tiles(img, cfg.tile_size, cfg.tile_overlap):
            dets = predictor.predict(tile.image, prompt, None, cfg.score_threshold)
            tiles_out.append((tile, _union(dets, tile.image.shape[:2])))
        return stitch_masks(tiles_out, img.shape[:2])
    dets = predictor.predict(img, prompt, None, cfg.score_threshold)
    return _union(dets, img.shape[:2])


def _union(dets: List[Detection], shape: Tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for d in dets:
        if d.mask.shape != shape:
            m = cv2.resize(
                d.mask.astype(np.uint8), (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            m = d.mask
        out |= m
    return out


def _color_prior_sky(img_rgb: np.ndarray) -> np.ndarray:
    """HSV sky detector: bright, low-to-medium saturation, blue hue.

    Catches clear blue sky and pale/white cloud areas near the top of the
    frame. Restricted to the top 60% of rows so bright patches on the
    glacier or meltwater don't get misclassified.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H = hsv.shape[0]
    # Blue sky: hue ~95–135, value high, saturation not-too-high
    blue = (
        (hsv[..., 0] >= 95) & (hsv[..., 0] <= 135) &
        (hsv[..., 1] <= 180) & (hsv[..., 2] >= 140)
    )
    # Bright pale clouds: very high value, very low saturation
    pale = (hsv[..., 1] <= 40) & (hsv[..., 2] >= 220)
    mask = blue | pale
    # Only the top 60% of the image
    cutoff = int(0.6 * H)
    mask[cutoff:] = False
    return mask


def _color_prior_yellow(img_rgb: np.ndarray) -> np.ndarray:
    """HSV yellow-placard detector: saturated yellow in the lower half.

    Tuned for the Greenland-style promotional placards / skateboards the
    researcher holds up in-frame: hue ~15–40, high saturation, high value.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H = hsv.shape[0]
    yellow = (
        (hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) &
        (hsv[..., 1] >= 120) & (hsv[..., 2] >= 150)
    )
    # Restrict to lower half — these props are always foreground.
    yellow[: H // 2] = False
    return yellow


def _postprocess_keep(mask: np.ndarray, cfg: AgentConfig) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = max(1, cfg.morph_close_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = m.astype(bool)
    if cfg.largest_component_only and m.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = labels == largest
    return m
