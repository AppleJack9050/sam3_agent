"""Fair evaluation of the retrained variants on the SAME 73 held-out views.

Every render is scored against the SAME reference: the unmasked photo resized exactly like the
training images (data/nomask/images, PIL LANCZOS 1000x750). Regions:
  full   : whole frame
  keep   : pixels kept by BOTH masks (agent and SAM3), eroded 4 px to drop the LANCZOS blend band
  removed: pixels removed by either mask (the sky) - shows what each model renders there
Metrics: PSNR (pooled over MSE and per-image mean), SSIM (torchmetrics gaussian SSIM map averaged
over the region), LPIPS-VGG (torchmetrics, GT pasted outside the region), RMSE.
Paired bootstrap (10k) over views for the differences between variants.
Writes eval/fair_metrics.json, eval/per_view.csv, eval/report.md
"""
import csv
import json
from pathlib import Path


import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

R = Path("/mnt/windows/Dataset/2016-11-28_Howchin-AlphLake_Imagery-Files.beh/retrain_fair")
V = ["nomask", "sam3_noagent", "agent"]
LABEL = {"nomask": "No mask (raw imagery)", "sam3_noagent": "SAM3, no agent", "agent": "Agent + validation"}
W, H = 1000, 750
ERODE = 4
dev = "cuda"

split = json.load(open(R / "runs" / "agent" / "split.json"))
for v in V:
    assert json.load(open(R / "runs" / v / "split.json")) == split, f"split mismatch {v}"
test = [Path(n).stem for n in split["test"]]


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)


def removed_small(stem):
    fr = []
    for m in ("agent", "sam3_sky"):
        a = Image.fromarray((np.asarray(Image.open(R / "masks" / m / f"{stem}.png")) > 0).astype(np.float32), mode="F")
        fr.append(np.asarray(a.resize((W, H), Image.BOX)))  # area average (4x4 -> 1)
    return np.maximum(fr[0], fr[1])  # fraction of each output pixel removed by either mask


def erode(mask, r):
    """Binary erosion with a (2r+1)^2 square via max-pool on the complement."""
    t = torch.from_numpy((~mask).astype(np.float32))[None, None]
    return (torch.nn.functional.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[0, 0] == 0).numpy()


lp = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(dev)
rows = []
agg = {v: {r: dict(se=0.0, n=0) for r in ("full", "keep", "removed")} for v in V}
for s in test:
    gt = load(R / "data" / "nomask" / "images" / f"{s}.png")
    rem = removed_small(s)
    keep = rem == 0
    if ERODE:
        keep = erode(keep, ERODE)
    removed = rem > 0.5
    km = torch.from_numpy(keep).to(dev)
    rm = torch.from_numpy(removed).to(dev)
    row = dict(stem=s, sky_frac=float(removed.mean()), keep_frac=float(keep.mean()))
    for v in V:
        pr = load(R / "runs" / v / "renders" / "test" / f"{s}.png")
        err = ((pr - gt) ** 2).mean(1)[0]  # HxW
        _, smap = ssim_fn(pr, gt, data_range=1.0, return_full_image=True)
        smap = smap.mean(1)[0]
        for reg, m in (("full", torch.ones_like(km)), ("keep", km), ("removed", rm)):
            n = int(m.sum())
            if n == 0:
                row[f"{v}_{reg}_psnr"] = row[f"{v}_{reg}_ssim"] = row[f"{v}_{reg}_lpips"] = row[f"{v}_{reg}_rmse"] = float("nan")
                continue
            mse = float(err[m].mean())
            agg[v][reg]["se"] += float(err[m].sum())
            agg[v][reg]["n"] += n
            row[f"{v}_{reg}_psnr"] = -10 * np.log10(max(mse, 1e-12))
            row[f"{v}_{reg}_rmse"] = float(np.sqrt(mse))
            row[f"{v}_{reg}_ssim"] = float(smap[m].mean())
            if reg != "removed":
                mixed = torch.where(m[None, None], pr, gt)
                row[f"{v}_{reg}_lpips"] = float(lp(mixed.clamp(0, 1), gt.clamp(0, 1)))
            else:
                row[f"{v}_{reg}_lpips"] = float("nan")
    rows.append(row)
    print(s, f"sky={row['sky_frac']:.3f}", " ".join(f"{v}:{row[f'{v}_keep_psnr']:.2f}" for v in V), flush=True)

out = R / "eval"
out.mkdir(exist_ok=True)
keys = list(rows[0].keys())
with open(out / "per_view.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

rng = np.random.default_rng(0)


def col(v, reg, met, sel=None):
    x = np.array([r[f"{v}_{reg}_{met}"] for r in rows], dtype=float)
    return x if sel is None else x[sel]


def boot(d):
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return [float("nan")] * 3
    idx = rng.integers(0, len(d), (10000, len(d)))
    bs = d[idx].mean(1)
    return [float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


sky = np.array([r["sky_frac"] > 0 for r in rows])
res = dict(n_views=len(rows), n_sky_views=int(sky.sum()), erode_px=ERODE, variants={}, paired={},
           naive_gsplat_eval={v: json.load(open(R / "runs" / v / "metrics.json")) for v in V})
for v in V:
    res["variants"][v] = {}
    for reg in ("full", "keep", "removed"):
        a = agg[v][reg]
        res["variants"][v][reg] = dict(
            psnr_pooled=-10 * np.log10(a["se"] / a["n"]) if a["n"] else None,
            psnr_mean=float(np.nanmean(col(v, reg, "psnr"))),
            ssim=float(np.nanmean(col(v, reg, "ssim"))),
            lpips=float(np.nanmean(col(v, reg, "lpips"))) if reg != "removed" else None,
            rmse=float(np.nanmean(col(v, reg, "rmse"))),
        )
for a, b in (("agent", "nomask"), ("agent", "sam3_noagent"), ("sam3_noagent", "nomask")):
    for reg in ("full", "keep"):
        for met in ("psnr", "ssim", "lpips"):
            for subset, sel in (("all", None), ("sky_views", sky), ("no_sky_views", ~sky)):
                res["paired"][f"{a}-{b}|{reg}|{met}|{subset}"] = boot(col(a, reg, met, sel) - col(b, reg, met, sel))
json.dump(res, open(out / "fair_metrics.json", "w"), indent=1)

L = ["# Fair re-evaluation (same 73 held-out views, same reference photos)\n",
     f"Views: {len(rows)} (with sky removed by the masks: {int(sky.sum())}). Keep region eroded {ERODE}px.\n",
     "| Variant | naive PSNR/SSIM/LPIPS (own GT, gsplat_eval) | Full frame PSNR / SSIM / LPIPS | Keep region PSNR / SSIM / LPIPS | Sky region PSNR |",
     "|---|---|---|---|---|"]
for v in V:
    nv = res["naive_gsplat_eval"][v]
    f_, k_, r_ = (res["variants"][v][x] for x in ("full", "keep", "removed"))
    L.append(f"| {LABEL[v]} | {nv['psnr']:.2f} / {nv['ssim']:.3f} / {nv['lpips']:.3f} | "
             f"{f_['psnr_pooled']:.2f} / {f_['ssim']:.3f} / {f_['lpips']:.3f} | "
             f"{k_['psnr_pooled']:.2f} / {k_['ssim']:.3f} / {k_['lpips']:.3f} | {r_['psnr_pooled']:.2f} |")
L.append("\nPSNR = pooled over all pixels of the region; SSIM/LPIPS = mean over views.\n")
L.append("## Paired differences (mean over views, 95% bootstrap CI)\n")
L.append("| Comparison | Region | Views | dPSNR (dB) | dSSIM | dLPIPS |")
L.append("|---|---|---|---|---|---|")
for a, b in (("agent", "nomask"), ("agent", "sam3_noagent"), ("sam3_noagent", "nomask")):
    for reg in ("keep", "full"):
        for subset in ("all", "sky_views", "no_sky_views"):
            c = [res["paired"][f"{a}-{b}|{reg}|{m}|{subset}"] for m in ("psnr", "ssim", "lpips")]
            L.append(f"| {LABEL[a]} - {LABEL[b]} | {reg} | {subset} | " + " | ".join(
                f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" for x in c) + " |")
open(out / "report.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
