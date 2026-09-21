# Glacier (Howchin / Alph Lake) — 3DGS results: with vs without the agent

Scene: 589 DJI drone photos, `/mnt/windows/Dataset/2016-11-28_Howchin-AlphLake_Imagery-Files.beh/`.
Trainer: `gsplat_eval` (github AppleJack9050/gsplat_eval), 30k iterations.

| Folder | What it is |
|---|---|
| `03_fair_retrain/` | **Use this.** All three variants retrained on equal terms, with rendered images. |
| `02_same_pose_original_models/` | Same-pose re-renders of the *original* (May 2026) models. |
| `01_original_table/` | The original runs behind the first table (kept for reference; not a valid comparison). |

## 03_fair_retrain (2026-09-13)

Setup, identical for all three variants: one shared COLMAP model (583 images), same 73 held-out
test views (every 8th), same preprocessing (original JPG → mask applied at full res → LANCZOS to
1000×750), `--downsample 4`, seed 42.

| Name | Training images |
|---|---|
| `no_mask` / `nomask` | original photos |
| `without_agent_sam3` / `sam3_noagent` | SAM3 only: `sam3_prompt`, prompt "sky", its default settings |
| `with_agent` / `agent` | sam3_agent masks (the exact masks behind the original "Ours" row) |

**Rendered images** — every image of the same name is the same camera pose, held out for every model:

- `figures/with_vs_without_agent.png` — left **without agent (SAM3)**, right **with agent**
- `figures/summary.png`, `summary_zoom.png`, `extremes_masking_effect.png` — columns:
  real photo | no mask | without agent (SAM3) | **with agent**
- `figures/per_view/DJI_xxxx.png` — all 73 views, 2×2 grid:
  top-left real photo, top-right no mask, bottom-left without agent (SAM3), **bottom-right with agent**
- `renders/with_agent/`, `renders/without_agent_sam3/`, `renders/no_mask/` — raw renders (1000×750)

**Results** (`eval/report.md`) — all renders scored against the same unmasked photos, on the region
both masks keep (terrain):

| Variant | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| No mask | 22.02 | 0.749 | 0.260 |
| Without agent (SAM3) | 23.06 | 0.754 | 0.256 |
| With agent | 23.02 | 0.753 | 0.256 |

- With agent − without agent: −0.02 dB [95% CI −0.13, +0.09] → no difference. The two mask sets are
  ~99% identical (IoU 0.987, same 269 sky frames).
- With agent − no mask: +0.30 dB [−0.02, +0.65]; masking removes sky floaters on some views
  (DJI_0571, 0563, 0179) but bakes black holes into distant mountains on others (DJI_0211, 0555).
- Single seed per variant.

Other files: `eval/per_view.csv`, `eval/fair_metrics.json`, `runs_info/<variant>/` (trainer metrics,
config, split, log), `masks_info/`, `scripts/` (make_masks → build_datasets → train_all → fair_eval →
make_figures). Full data, masks and checkpoints (~11 GB) stay in `.../retrain_fair/` on the data drive.

## 02_same_pose_original_models

The original models' test views don't overlap, so these re-render each original checkpoint at the
same photo poses and at midpoint poses between adjacent frames (unseen by all models).
`summary_agent_vs_noagent_midpoint.png` (left without agent, right with agent, plus zooms) and
`pose_compare_*.png` (real photo | no mask | SAM3 row | agent).

## 01_original_table — why it is not a valid comparison

Original numbers: No mask 21.74 / SAM3 no agent 21.66 / Ours 22.22 dB. Problems found:
- the "SAM3, no agent" run trained on the unmasked photos (its SAM3 output went to a folder the trainer never read);
- "Ours" used a different image preprocessing (2048 px area-resize), which alone adds ~+1.3 dB;
- disjoint test views, separate COLMAP per run, and "Ours" scored against its own black-sky images.
