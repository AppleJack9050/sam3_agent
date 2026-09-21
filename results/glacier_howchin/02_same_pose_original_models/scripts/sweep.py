"""Render every view registered in all 3 COLMAP models with all 3 checkpoints at one common
resolution, and score each render against the SAME real photo (unmasked JPG), on the full frame
and inside the agent keep-region. Writes sweep.csv. Read-only on the source data."""
import csv
import json
import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from gsrender import B, S, read_colmap_txt, intrinsics, load_ply, render

W, H = 500, 375  # common eval resolution (== unmasked/native_prompt eval res; masked evaluated at 512x384)
VARIANTS = ["unmasked", "native_prompt", "masked"]

poses = {v: read_colmap_txt(v) for v in VARIANTS}
common = sorted(set.intersection(*(set(poses[v][1]) for v in VARIANTS)))
stem = lambda n: n.rsplit(".", 1)[0]
test = {v: set(map(stem, json.load(open(f"{B}/{v}/runs/my_experiment/split.json"))["test"])) for v in VARIANTS}
print("common registered views:", len(common), flush=True)

gt_cache = f"{S}/gt_{W}x{H}.npz"
if os.path.exists(gt_cache):
    z = np.load(gt_cache)
    GT, KEEP, GTM = z["gt"], z["keep"], z["gtm"]
else:
    GT = np.zeros((len(common), H, W, 3), np.uint8)
    GTM = np.zeros((len(common), H, W, 3), np.uint8)
    KEEP = np.zeros((len(common), H, W), bool)
    for i, s in enumerate(common):
        im = Image.open(f"{B}/unmasked/images/{s}.JPG")
        im.draft("RGB", (W * 2, H * 2))
        GT[i] = np.asarray(im.convert("RGB").resize((W, H), Image.LANCZOS))
        m = Image.open(f"{B}/masked/images/{s}.png").convert("RGB")
        ma = np.asarray(m)
        keep_full = (ma.max(-1) > 0).astype(np.float32)
        KEEP[i] = np.asarray(Image.fromarray(keep_full).resize((W, H), Image.BOX)) > 0.5
        GTM[i] = np.asarray(m.resize((W, H), Image.LANCZOS))
        if i % 100 == 0:
            print("  gt", i, flush=True)
    np.savez(gt_cache, gt=GT, keep=KEEP, gtm=GTM)

rows = {s: dict(stem=s, keep_frac=float(KEEP[i].mean())) for i, s in enumerate(common)}
for i, s in enumerate(common):
    k = KEEP[i]
    g = GT[i].astype(np.float32) / 255
    gm = GTM[i].astype(np.float32) / 255
    # how well the agent's masked training photo agrees with the unmasked photo inside keep region
    rows[s]["gtm_vs_gt_psnr_keep"] = float(-10 * np.log10(max(((gm - g)[k] ** 2).mean(), 1e-12))) if k.any() else float("nan")

for v in VARIANTS:
    cams, imgs = poses[v]
    model = load_ply(f"{B}/{v}/runs/my_experiment/checkpoints/point_cloud_30000.ply")
    for i, s in enumerate(common):
        im = imgs[s]
        out = render(model, im["R"], im["t"], intrinsics(cams[im["cam"]], W, H), W, H)
        g = GT[i].astype(np.float32) / 255
        err = (out - g) ** 2
        k = KEEP[i]
        rows[s][f"{v}_split"] = "test" if s in test[v] else "train"
        rows[s][f"{v}_psnr_full"] = float(-10 * np.log10(max(err.mean(), 1e-12)))
        rows[s][f"{v}_psnr_keep"] = float(-10 * np.log10(max(err[k].mean(), 1e-12))) if k.any() else float("nan")
        rows[s][f"{v}_psnr_excl"] = float(-10 * np.log10(max(err[~k].mean(), 1e-12))) if (~k).any() else float("nan")
    print(v, "rendered", len(common), flush=True)
    del model
    torch.cuda.empty_cache()

keys = list(next(iter(rows.values())).keys())
with open(f"{S}/sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for s in common:
        w.writerow(rows[s])
print("wrote", f"{S}/sweep.csv")
