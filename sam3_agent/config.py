from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


@dataclass
class AgentConfig:
    # --- Strategy -------------------------------------------------------
    # "exclude" (default): SAM 3 segments sky / distant background /
    #   foreground props. We union those, invert, and keep the complement.
    #   Best for ice-cliff / terminus scenes where the boundary we want is
    #   the ridge silhouette against the sky.
    # "include": SAM 3 segments the glacier ice directly and we keep that
    #   mask. Better for straight-down aerial ice-vs-sediment shots.
    mode: Literal["include", "exclude"] = "exclude"

    # Human-readable description of what we keep; used for LLM reasoning.
    preserve_label: str = (
        "glacier ice wall, exposed rock and sediment layers, moraine "
        "deposits, foreground terrain, glacial meltwater pools — everything "
        "up to but not including the sky and distant background"
    )

    # Classes to black out (exclude-mode default).
    exclude_prompts: Tuple[str, ...] = field(
        default_factory=lambda: (
            "sky",
            "clouds",
            "distant mountains and background landscape beyond the glacier",
            "far horizon, snow on far peaks",
            "buildings, vehicles, people, tents, antennas, wires, flags",
            "handheld signs, placards, skateboards, promotional cards, stickers",
        )
    )

    # Alternative exclude prompt sets cycled through on quality failure.
    exclude_prompt_fallbacks: Tuple[Tuple[str, ...], ...] = field(
        default_factory=lambda: (
            (
                "sky", "clouds", "hazy atmosphere",
                "snow-covered far peaks above the glacier ridge",
                "sun and lens flare", "aircraft",
                "any man-made foreground object",
            ),
            (
                "bright sky and thin clouds",
                "distant snowy ridges far beyond the glacier",
                "horizon line",
                "yellow or red handheld signs in the foreground",
            ),
        )
    )

    # --- Include-mode prompts ------------------------------------------
    # SAM 3 is queried once per entry; masks are unioned.
    include_prompts: Tuple[str, ...] = field(
        default_factory=lambda: (
            "glacier ice",
            "ice surface with crevasses",
            "dirty ice with sediment stripes",
            "frost and snow on ice",
            "meltwater pool on ice",
        )
    )

    # Fallback prompt sets cycled through on quality failure.
    include_prompt_fallbacks: Tuple[Tuple[str, ...], ...] = field(
        default_factory=lambda: (
            ("glacier ice, including dirty ice and crevassed ice",),
            ("any ice surface, bluish or white",
             "meltwater-soaked ice",
             "ice with dust and sediment layers"),
        )
    )

    # Single-prompt override for the legacy `--preserve` flag.
    target: Optional[str] = None

    exemplar_image: Optional[str] = None

    # --- Preprocessing -------------------------------------------------
    target_long_side: int = 2048
    color_space: str = "RGB"

    # --- Tiling --------------------------------------------------------
    tile_size: int = 1024
    tile_overlap: int = 128
    tile_threshold: int = 2048

    # --- SAM3 ----------------------------------------------------------
    model_name: str = "facebook/sam3"
    device: str = "cuda"
    cuda_arch: str = "12.0"
    score_threshold: float = 0.3

    # --- Color-based priors (augment SAM 3 exclude mask) ---------------
    # Opt-in only. These HSV priors are tuned for ground-level shots with a
    # visible sky strip and/or a saturated yellow placard. They MISFIRE on
    # aerial ice imagery (bluish ice looks like sky in HSV), so default off.
    # Enable from the CLI with --color-prior-sky / --color-prior-yellow.
    color_prior_sky: bool = False
    color_prior_yellow: bool = False

    # --- Post-processing on the keep mask ------------------------------
    largest_component_only: bool = True
    morph_close_px: int = 7
    # Outward dilation (px) of the final keep mask. A small dilation makes the
    # kept region a slight SUPERSET of the true foreground, which markedly improves
    # downstream 3D reconstruction (Gaussian Splatting): every shared-foreground 3D
    # point is then kept in *all* views instead of being blacked-out (and averaged
    # toward black) in some, and the black/foreground blend boundary introduced by
    # downsampling sits outside the object interior. 0 disables. Applied at the mask
    # resolution, so dilate_px scales with target_long_side / full-res output.
    dilate_px: int = 0

    # --- Output resolution ---------------------------------------------
    # When True (default), the keep mask is upscaled back to the ORIGINAL image
    # resolution and applied to the full-res image, so the saved masked image
    # keeps native-resolution pixels with a hard mask edge. When False (legacy),
    # the mask is applied to the `target_long_side`-downscaled image, which both
    # softens the kept pixels and leaves a resampling halo at the mask boundary
    # (harmful for downstream 3D reconstruction / Gaussian Splatting).
    output_full_resolution: bool = True

    # --- include-mode instance selection -------------------------------
    # 0 (default): keep the UNION of every SAM 3 detection above score_threshold
    #   across all include prompts (good for multi-part targets, e.g. glacier ice
    #   queried with several prompts).
    # k>0: keep only the top-k highest-confidence detections (pooled across
    #   prompts). With a single `--preserve` target this avoids over-segmentation
    #   — e.g. grabbing a bench a bicycle rests on — and yields tighter, more
    #   multi-view-consistent masks. Ignored in exclude mode and when tiling.
    include_top_k: int = 0

    # --- Quality gates (phrased on the *keep* mask) --------------------
    # Defaults loosened: an aerial shot of a glacier tongue can legitimately
    # be 90%+ ice or as low as 30% ice depending on framing.
    min_keep_coverage: float = 0.05
    max_keep_coverage: float = 0.98
    max_retries: int = 3

    # --- Output --------------------------------------------------------
    output_format: str = "png"  # png | jpeg
    save_raw_mask: bool = True

    # --- LLM-assisted orchestration (Claude Agent SDK) -----------------
    # When use_llm is True, the adapt step asks Claude (via the Claude Agent
    # SDK) for the next prompt set. Auth uses your Claude *subscription* (the
    # SDK shells out to the logged-in `claude` CLI); keep ANTHROPIC_API_KEY
    # unset so billing goes to your plan, not the metered API. Any SDK / CLI /
    # auth failure falls back to deterministic prompt cycling.
    use_llm: bool = False
    # Model id or alias ("opus" / "sonnet" / "haiku") passed to the SDK.
    llm_model: str = "claude-opus-4-8"
    # Optional per-call cost ceiling (USD) for the Agent SDK; None = no cap.
    llm_max_budget_usd: Optional[float] = None

    def __post_init__(self):
        # Backwards compatibility: AgentConfig(target="glacier") → include mode
        # with that single prompt.
        if self.target is not None and self.mode == "exclude":
            self.mode = "include"
