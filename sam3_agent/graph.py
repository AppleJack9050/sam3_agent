"""LangGraph state machine for the SAM 3 segment-by-exclusion agent.

Nodes:
    preprocess → build_prompts → sam3_infer → quality ─ ok → save → END
                                     ▲             │
                                     └─ adapt ◀─ fail (retry)

In `exclude` mode (default) the graph queries SAM 3 once per prompt in
`cfg.exclude_prompts`, unions the masks, inverts, morph-closes, and keeps the
largest connected component. In `include` mode it keeps SAM 3's positive mask
directly (legacy behavior).
"""
from __future__ import annotations

import json
from typing import List, Literal, Optional, Tuple, TypedDict

import cv2
import numpy as np
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .config import AgentConfig
from .input_module import (
    Tile,
    iter_tiles,
    load_image,
    normalize_resolution,
    should_tile,
    stitch_masks,
)
from .output import save_result
from .quality import QualityReport, assess
from .sam3_inference import Detection, SAM3Predictor


# ---------- Graph state ----------

class AgentState(TypedDict, total=False):
    image_path: str
    output_path: str
    cfg: AgentConfig

    image: np.ndarray
    exemplar: Optional[np.ndarray]

    prompts: List[str]
    attempt: int

    keep_mask: np.ndarray
    exclude_mask: np.ndarray
    quality: QualityReport

    output_paths: dict
    history: List[dict]


# ---------- LLM structured output ----------

class PromptSetDecision(BaseModel):
    prompts: List[str] = Field(description="Next exclude-mode prompt set.")
    reasoning: str = Field(description="Brief reasoning.")


# ---------- Graph ----------

def build_graph(predictor: SAM3Predictor):
    def node_preprocess(state: AgentState) -> AgentState:
        cfg: AgentConfig = state["cfg"]
        img = load_image(state["image_path"], color_space=cfg.color_space)
        img, _ = normalize_resolution(img, cfg.target_long_side)
        exemplar = (
            load_image(cfg.exemplar_image, color_space=cfg.color_space)
            if cfg.exemplar_image else None
        )
        return {**state, "image": img, "exemplar": exemplar, "attempt": 0, "history": []}

    def node_build_prompts(state: AgentState) -> AgentState:
        cfg: AgentConfig = state["cfg"]
        if state.get("prompts"):
            return state  # already set by adapt_prompts on retry
        if cfg.mode == "include":
            if cfg.target:
                prompts = [cfg.target]  # single-prompt override
            else:
                prompts = list(cfg.include_prompts)
        else:
            prompts = list(cfg.exclude_prompts)
        return {**state, "prompts": prompts}

    def node_sam3_infer(state: AgentState) -> AgentState:
        cfg: AgentConfig = state["cfg"]
        img = state["image"]
        prompts = state["prompts"]

        raw = np.zeros(img.shape[:2], dtype=bool)
        for prompt in prompts:
            raw |= _infer_one(predictor, img, prompt, cfg)

        if cfg.mode == "exclude":
            # Augment SAM 3's output with deterministic color priors so the
            # sky and yellow placards are excluded even when SAM 3 misses them.
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
        return {
            **state,
            "keep_mask": keep,
            "exclude_mask": exclude,
            "attempt": state.get("attempt", 0) + 1,
        }

    def node_quality(state: AgentState) -> AgentState:
        cfg: AgentConfig = state["cfg"]
        report = assess(
            state["keep_mask"],
            state.get("exclude_mask"),
            state["image"],
            cfg.min_keep_coverage,
            cfg.max_keep_coverage,
        )
        hist = list(state.get("history", []))
        hist.append({
            "attempt": state["attempt"],
            "mode": cfg.mode,
            "prompts": list(state["prompts"]),
            "coverage": report.coverage,
            "ok": report.ok,
            "reasons": report.reasons,
        })
        return {**state, "quality": report, "history": hist}

    def node_adapt_prompts(state: AgentState) -> AgentState:
        cfg: AgentConfig = state["cfg"]
        decision = _decide_next(
            cfg=cfg,
            attempt=state["attempt"],
            last_prompts=state["prompts"],
            reasons=state["quality"].reasons,
        )
        return {**state, "prompts": decision.prompts}

    def node_save(state: AgentState) -> AgentState:
        cfg: AgentConfig = state["cfg"]
        paths = save_result(
            state["image"], state["keep_mask"], state["output_path"],
            fmt=cfg.output_format, save_raw_mask=cfg.save_raw_mask,
        )
        return {**state, "output_paths": paths}

    def route_after_quality(state: AgentState) -> Literal["adapt", "save"]:
        cfg: AgentConfig = state["cfg"]
        if state["quality"].ok:
            return "save"
        if state["attempt"] >= cfg.max_retries:
            return "save"
        return "adapt"

    g = StateGraph(AgentState)
    g.add_node("preprocess", node_preprocess)
    g.add_node("build_prompts", node_build_prompts)
    g.add_node("sam3_infer", node_sam3_infer)
    g.add_node("quality", node_quality)
    g.add_node("adapt_prompts", node_adapt_prompts)
    g.add_node("save", node_save)
    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "build_prompts")
    g.add_edge("build_prompts", "sam3_infer")
    g.add_edge("sam3_infer", "quality")
    g.add_conditional_edges("quality", route_after_quality, {"adapt": "adapt_prompts", "save": "save"})
    g.add_edge("adapt_prompts", "sam3_infer")
    g.add_edge("save", END)
    return g.compile()


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


def _decide_next(
    cfg: AgentConfig,
    attempt: int,
    last_prompts: List[str],
    reasons: List[str],
) -> PromptSetDecision:
    fallbacks = list(
        cfg.include_prompt_fallbacks if cfg.mode == "include"
        else cfg.exclude_prompt_fallbacks
    )
    if cfg.use_llm:
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.prompts import ChatPromptTemplate

            llm = ChatAnthropic(model=cfg.llm_model, max_tokens=500, temperature=0)
            structured = llm.with_structured_output(PromptSetDecision)
            tmpl = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You orchestrate a SAM 3 segment-by-exclusion agent for glacier-"
                    "terminus scenes. The agent queries SAM 3 for each prompt in a list "
                    "of *exclude* classes (sky, clouds, far mountains, man-made objects, "
                    "etc.), unions the masks, and inverts to keep the foreground. Given "
                    "the last prompt set and the reasons it failed quality checks, "
                    "propose the next prompt set. Keep prompts as short noun phrases.",
                ),
                (
                    "user",
                    "Preserve target: {preserve}\nAttempt: {attempt}\n"
                    "Last exclude prompts: {last}\nQuality failures: {reasons}\n"
                    "Hint prompt sets: {hints}",
                ),
            ])
            decision = (tmpl | structured).invoke({
                "preserve": cfg.preserve_label,
                "attempt": attempt,
                "last": json.dumps(last_prompts),
                "reasons": json.dumps(reasons),
                "hints": json.dumps([list(h) for h in fallbacks]),
            })
            return PromptSetDecision(prompts=list(decision.prompts), reasoning=decision.reasoning)
        except Exception:
            pass

    idx = min(attempt - 1, len(fallbacks) - 1) if fallbacks else 0
    prompts = list(fallbacks[idx]) if fallbacks else last_prompts
    return PromptSetDecision(prompts=prompts, reasoning="deterministic fallback")
