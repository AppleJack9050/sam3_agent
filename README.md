# SAM 3 Segmentation Agent (Claude Agent SDK)

An AI-agent wrapper around **SAM 3** for glacier-terminus imagery.
Orchestration is a deterministic Python pipeline; LLM-assisted prompt
refinement uses the **Claude Agent SDK** (`claude-agent-sdk`) with JSON-schema
structured output, authenticated against your **Claude subscription** (not a
metered API key).

## Strategy: segment by exclusion

Instead of asking SAM 3 to find the *glacier* (which misses moraine, rock
layers, and meltwater), the agent asks SAM 3 to find what we want to **remove**:

```
sky, clouds, distant mountains in the background, horizon,
buildings/vehicles/people/antennas/wires/flags
```

The agent unions those masks, **inverts** them, morph-closes the result, and
keeps the largest connected component. What remains is the glacier-terminus
scene — ice cliff face, exposed sediment/rock, moraine debris, foreground
gravel, and meltwater pools — with everything above the ridge silhouette
blacked out. The ridge silhouette is, by construction, the top edge of the
exclusion masks.

An `include` mode is available via `--mode include` for scenes where the
positive concept is easier to enumerate than the background — e.g.
straight-down aerial shots of ice-on-sediment where there's no sky at all.
Default exclude mode is the right choice for ground-level and oblique
aerial shots of a glacier terminus / ice cliff.

## Architecture

```
    ┌─ preprocess ─ build_prompts ─ sam3_infer ─ quality ─┐
    │                                   ▲    (fail)      │
    │                                   │                ▼
    │                                   └── adapt_prompts  (LLM or fallback)
    │                                                    │
    └─────────────── (ok or retries exhausted) ── save → END
```

| Module | Role |
|---|---|
| `pipeline.py` | deterministic retry loop — multi-prompt fan-out, invert, postprocess, retry |
| `llm.py` | **Claude Agent SDK** prompt adaptation — structured output, subscription auth |
| `input_module.py` | load, color-space normalize, resolution normalize, tile + stitch |
| `sam3_inference.py` | SAM 3 wrapper (`facebookresearch/sam3` image model) + prompt-dispatching mock |
| `quality.py` | keep-coverage gate + top-edge-hits-frame-top + sky-color leakage |
| `agent.py` | thin facade: runs the pipeline |
| `output.py` | PNG or JPEG with pure black outside the mask + raw mask PNG |
| `api.py` | FastAPI service: `/segment`, `/segment/batch` |
| `cli.py` | CLI entrypoint |

The adapt step (`llm.decide_next_prompts`) calls the Claude Agent SDK's
`query()` with a JSON-schema `output_format`, so Claude returns a typed
decision (`prompts`, `reasoning`) billed to your subscription. If the SDK, the
`claude` CLI, or the login is unavailable — or `use_llm` is off — it falls back
to cycling through `AgentConfig.exclude_prompt_fallbacks`.

## Install

**Hardware target:** NVIDIA **RTX 5090** (Blackwell, compute capability **sm_120**).
Blackwell requires **CUDA 12.8+** and **PyTorch ≥ 2.7** built with `sm_120` kernels.

```bash
# 1. Create the conda environment (Python only — everything else is pip).
conda create -n sam3-agent python=3.11 -y
conda activate sam3-agent

# 2. Install PyTorch with Blackwell (sm_120) support from the cu128 wheel index.
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7" torchvision

# 3. Install the agent's dependencies (incl. claude-agent-sdk) and SAM 3.
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam3.git

# 4. Install the Claude Code CLI — the Claude Agent SDK shells out to it — and
#    log in with your Claude subscription. The OAuth login is what bills your
#    plan instead of the metered API.
npm install -g @anthropic-ai/claude-code   # provides the `claude` binary on PATH
claude login                                # interactive; or `claude setup-token` for headless/CI

# IMPORTANT: keep ANTHROPIC_API_KEY UNSET, or the SDK/CLI will bill the metered
# API instead of your subscription. For headless use, export the token from
# `claude setup-token` as CLAUDE_CODE_OAUTH_TOKEN.
unset ANTHROPIC_API_KEY
```

> Only the LLM-assisted `--use-llm` path needs the Claude Code CLI and login.
> The default deterministic pipeline (and all tests, which use the mock SAM 3
> predictor) runs without it.

Verify the CUDA backend is live on the 5090:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expected: True  NVIDIA GeForce RTX 5090  (12, 0)
```

SAM 3 has 848M parameters; RTX 5090 (32 GB VRAM, Blackwell) handles full-resolution
inference comfortably.

## Use — CLI

```bash
# Default (exclude mode) — black out sky, far mountains, man-made objects.
python -m sam3_agent /path/to/glacier.jpg out/glacier --format png

# Override the exclude list
python -m sam3_agent img.jpg out/res \
    --exclude sky --exclude clouds --exclude "snow on distant peaks"

# Legacy include mode (keep SAM 3's positive mask directly)
python -m sam3_agent img.jpg out/res --mode include --preserve glacier

# Mock mode (no GPU; the mock dispatches on prompt keywords)
python -m sam3_agent sample.jpg out/result --mock

# Batch mode — JSONL manifest with one {"input": "...", "output": "..."} per line
python -m sam3_agent --batch manifest.jsonl
```

Manifest format:

```jsonl
{"input": "img1.jpg", "output": "out/img1"}
{"input": "img2.tif", "output": "out/img2"}
```

## Use — Python

```python
from sam3_agent import SAM3Agent, AgentConfig

agent = SAM3Agent(AgentConfig(
    mode="exclude",
    exclude_prompts=("sky", "clouds", "distant mountains", "buildings"),
    use_llm=True,
))
result = agent.run("glacier.jpg", "out/glacier")
print(result.coverage, result.attempts, result.final_prompts)
```

## Use — REST

```bash
uvicorn sam3_agent.api:app --host 0.0.0.0 --port 8000

# Default exclude mode
curl -F file=@glacier.jpg \
     -F mode=exclude \
     -F exclude_prompts='sky,clouds,distant mountains,buildings' \
     http://localhost:8000/segment

# Include mode
curl -F file=@glacier.jpg -F mode=include -F preserve=glacier \
     http://localhost:8000/segment

# Batch (JSON body)
curl -X POST http://localhost:8000/segment/batch \
     -H 'content-type: application/json' \
     -d '{"items":[{"input_path":"a.jpg","output_path":"out/a"}],"mode":"exclude"}'
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

Tests run entirely on CPU using `MockSAM3Predictor` — no SAM 3 weights or
network needed.

## Agent control flow

1. Load image → normalize resolution → (optional) tile if max side > threshold.
2. Build prompt (default: target noun phrase, e.g. `"glacier"`).
3. SAM 3 inference per tile; union masks; stitch to full resolution.
4. Quality check: coverage ∈ [min, max], no "ship-like" false-positive cluster.
5. On failure: Claude (via the Agent SDK, on your subscription) proposes the
   next prompt set, or deterministic fallback cycling when `use_llm` is off or
   the SDK is unavailable; then retry (up to `max_retries`).
6. Blackout pixels outside mask and save PNG (alpha) or JPEG (+ raw mask PNG).

## Tiling for large remote-sensing imagery

If `max(H, W) > tile_threshold` (default 2048), the agent slices the image into
overlapping `tile_size × tile_size` patches (default 1024 / 128 overlap), runs
SAM 3 per tile, and stitches the masks via boolean OR in overlap regions. Edge
tiles are shifted inward so every tile is full-sized.

## Swapping in the real SAM 3 API

`SAM3Predictor` in `sam3_inference.py` tries the official `sam3` Python API
first, then falls back to the Ultralytics integration. Adjust the exact import
names to match the SAM 3 release you install — the orchestration layer doesn't
need to change.
