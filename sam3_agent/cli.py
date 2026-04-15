"""CLI entrypoint.

Default (exclude mode): black out sky, clouds, far mountains, and man-made
objects; keep the glacier-terminus scene.

    python -m sam3_agent path/to/image.jpg out/result --format png

Override the exclude list:

    python -m sam3_agent img.jpg out/res --exclude sky --exclude clouds

Legacy include mode (keep SAM 3's positive mask):

    python -m sam3_agent img.jpg out/res --mode include --preserve glacier

Batch (JSONL with one {"input": "...", "output": "..."} per line):

    python -m sam3_agent --batch manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Tuple

from .agent import SAM3Agent
from .config import AgentConfig
from .sam3_inference import MockSAM3Predictor, SAM3Predictor


def _read_manifest(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                pairs.append((obj["input"], obj["output"]))
            except KeyError as e:
                raise SystemExit(f"{path}:{line_no}: missing key {e}") from e
    if not pairs:
        raise SystemExit(f"{path}: no entries")
    return pairs


def main():
    p = argparse.ArgumentParser(description="SAM 3 segment-by-exclusion agent")
    p.add_argument("input", nargs="?", help="input image path (omit when --batch)")
    p.add_argument("output", nargs="?", help="output path, no extension (omit when --batch)")
    p.add_argument("--batch", help="JSONL manifest with {input, output} per line")

    p.add_argument("--mode", choices=["exclude", "include"], default="exclude")
    p.add_argument(
        "--exclude", action="append", default=None,
        help="exclude prompt (repeatable). Overrides the default exclude list.",
    )
    p.add_argument(
        "--preserve", default=None,
        help="in --mode include, the positive SAM 3 prompt; in --mode exclude, a label only.",
    )

    p.add_argument("--format", default="png", choices=["png", "jpeg", "jpg"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--mock", action="store_true", help="use heuristic mock predictor (no GPU)")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--color-prior-sky", action="store_true",
                   help="enable HSV sky prior (ground-level shots only; misfires on aerial ice)")
    p.add_argument("--color-prior-yellow", action="store_true",
                   help="enable HSV yellow-placard prior")
    args = p.parse_args()

    if args.batch:
        if args.input or args.output:
            p.error("--batch is mutually exclusive with positional input/output")
        pairs = _read_manifest(args.batch)
    else:
        if not args.input or not args.output:
            p.error("input and output are required unless --batch is given")
        pairs = [(args.input, args.output)]

    cfg_kwargs = dict(
        mode=args.mode,
        output_format=args.format,
        device=args.device,
        use_llm=args.use_llm,
        tile_size=args.tile_size,
        max_retries=args.max_retries,
        color_prior_sky=args.color_prior_sky,
        color_prior_yellow=args.color_prior_yellow,
    )
    if args.exclude:
        cfg_kwargs["exclude_prompts"] = tuple(args.exclude)
    if args.preserve:
        if args.mode == "include":
            cfg_kwargs["target"] = args.preserve
        else:
            cfg_kwargs["preserve_label"] = args.preserve

    cfg = AgentConfig(**cfg_kwargs)
    predictor = MockSAM3Predictor() if args.mock else SAM3Predictor(cfg.model_name, cfg.device)
    agent = SAM3Agent(cfg, predictor=predictor)

    results = agent.run_batch(pairs)
    json.dump(
        [
            {
                "input": pairs[i][0],
                "mode": r.mode,
                "output_paths": r.output_paths,
                "coverage": r.coverage,
                "attempts": r.attempts,
                "final_prompts": r.final_prompts,
                "quality_ok": r.quality.ok,
                "quality_reasons": r.quality.reasons,
            }
            for i, r in enumerate(results)
        ],
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
