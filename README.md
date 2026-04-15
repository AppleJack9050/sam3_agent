# SAM 3 Segmentation Agent (LangGraph)

An AI-agent wrapper around **SAM 3** for glacier-terminus imagery.
Orchestration is a **LangGraph** state machine; LLM-assisted prompt refinement
uses **LangChain**'s `ChatAnthropic` with structured output.

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
| `graph.py` | **LangGraph** `StateGraph` — multi-prompt fan-out, invert, postprocess, retry |
| `input_module.py` | load, color-space normalize, resolution normalize, tile + stitch |
| `sam3_inference.py` | SAM 3 wrapper (`facebookresearch/sam3` image model) + prompt-dispatching mock |
| `quality.py` | keep-coverage gate + top-edge-hits-frame-top + sky-color leakage |
| `agent.py` | thin facade: compiles the graph and invokes it |
| `output.py` | PNG or JPEG with pure black outside the mask + raw mask PNG |
| `api.py` | FastAPI service: `/segment`, `/segment/batch` |
| `cli.py` | CLI entrypoint |

The `adapt_prompts` node calls `ChatAnthropic(...).with_structured_output(PromptSetDecision)`
so the LLM returns a typed Pydantic decision (`prompts`, `reasoning`); if
LangChain / the API key is unavailable it cycles through
`AgentConfig.exclude_prompt_fallbacks`.

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

# 3. Install the agent's dependencies and SAM 3.
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam3.git
```

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
5. On failure: either deterministic fallback cycling or Claude picks the next
   prompt / flips the exemplar flag, then retries (up to `max_retries`).
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
