"""Rendered-image comparisons for the retrained variants. All 73 views are held out for ALL models
and share the exact same camera pose (one shared COLMAP model), so panels are directly comparable.

  figures/per_view/<stem>.png  : Real photo | No mask | SAM3, no agent | Agent + validation (all 73)
  figures/summary.png          : 4 rule-picked views (sky views at the 25/50/75th percentile of the
                                 agent-minus-SAM3 keep-region PSNR, + the median no-sky view)
  figures/summary_zoom.png     : same views, 2x zoom crops: ridge (mask boundary) and the window of
                                 largest disagreement between the three renders inside the kept region
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path("/mnt/windows/Dataset/2016-11-28_Howchin-AlphLake_Imagery-Files.beh/retrain_fair")
V = ["nomask", "sam3_noagent", "agent"]
LABEL = {"nomask": "No mask (raw imagery)", "sam3_noagent": "SAM3, no agent", "agent": "Agent + validation (ours)"}
W, H = 1000, 750
FIG = R / "figures"
(FIG / "per_view").mkdir(parents=True, exist_ok=True)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
FR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)

rows = {r["stem"]: r for r in csv.DictReader(open(R / "eval" / "per_view.csv"))}
f = lambda s, k: float(rows[s][k])


def img(p):
    return np.asarray(Image.open(p).convert("RGB"))


def photo_with_edges(s):
    ph = img(R / "data" / "nomask" / "images" / f"{s}.png").copy()
    for m, color in (("agent", (255, 40, 40)), ("sam3_sky", (40, 200, 255))):
        rm = (np.asarray(Image.open(R / "masks" / m / f"{s}.png")) > 0).astype(np.uint8)
        rm = cv2.resize(rm, (W, H), interpolation=cv2.INTER_NEAREST)
        edge = cv2.morphologyEx(rm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        ph[edge] = color
    return ph


def caption(pil, text, font=FR, h=34):
    d = ImageDraw.Draw(pil)
    d.rectangle([0, pil.height - h, pil.width, pil.height], fill=(0, 0, 0))
    d.text((10, pil.height - h + 5), text, fill=(255, 255, 255), font=font)
    return pil


def metrics_line(s, v):
    return f"{LABEL[v]}   keep-region PSNR {f(s, f'{v}_keep_psnr'):.2f} dB  SSIM {f(s, f'{v}_keep_ssim'):.3f}  LPIPS {f(s, f'{v}_keep_lpips'):.3f}"


# ---- per-view 2x2 grids (full resolution) ----
for s in rows:
    panels = [caption(Image.fromarray(photo_with_edges(s)), f"{s} real photo (held out)   red = agent mask edge, blue = SAM3 mask edge")]
    for v in V:
        panels.append(caption(Image.open(R / "runs" / v / "renders" / "test" / f"{s}.png").convert("RGB"), metrics_line(s, v)))
    grid = Image.new("RGB", (2 * W, 2 * H), (255, 255, 255))
    for i, p in enumerate(panels):
        grid.paste(p, ((i % 2) * W, (i // 2) * H))
    grid.save(FIG / "per_view" / f"{s}.png")

# ---- rule-based selection ----
sky = sorted((f(s, "agent_keep_psnr") - f(s, "sam3_noagent_keep_psnr"), s) for s in rows if f(s, "sky_frac") > 0)
nosky = sorted((f(s, "agent_keep_psnr") - f(s, "nomask_keep_psnr"), s) for s in rows if f(s, "sky_frac") == 0)
pick = [sky[int(round(q * (len(sky) - 1)))][1] for q in (0.25, 0.5, 0.75)] + [nosky[len(nosky) // 2][1]]
print("picked:", pick)

PW, PH = 640, 480
cols = ["Real photo (held out)"] + [LABEL[v] for v in V]
HEAD = 44


def summary(get_panel, name, title_suffix=""):
    fig = Image.new("RGB", (PW * 4, HEAD + PH * len(pick)), (255, 255, 255))
    d = ImageDraw.Draw(fig)
    for i, c in enumerate(cols):
        d.text((i * PW + 10, 9), c + title_suffix, fill=(0, 0, 0), font=FB)
    for r, s in enumerate(pick):
        for c in range(4):
            fig.paste(get_panel(s, c), (c * PW, HEAD + r * PH))
    fig.save(FIG / name)
    print("wrote", FIG / name)


def full_panel(s, c):
    if c == 0:
        p = Image.fromarray(photo_with_edges(s)).resize((PW, PH), Image.LANCZOS)
        return caption(p, f"{s}  sky removed {f(s, 'sky_frac') * 100:.0f}%")
    v = V[c - 1]
    p = Image.open(R / "runs" / v / "renders" / "test" / f"{s}.png").convert("RGB").resize((PW, PH), Image.LANCZOS)
    return caption(p, f"keep PSNR {f(s, f'{v}_keep_psnr'):.2f}  SSIM {f(s, f'{v}_keep_ssim'):.3f}  LPIPS {f(s, f'{v}_keep_lpips'):.3f}")


summary(full_panel, "summary.png")

CW, CH = 320, 240  # crop in 1000x750 pixels, shown 2x


def crops_for(s):
    rend = [img(R / "runs" / v / "renders" / "test" / f"{s}.png").astype(np.int16) for v in V]
    rm = np.maximum(*[cv2.resize((np.asarray(Image.open(R / "masks" / m / f"{s}.png")) > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) for m in ("agent", "sam3_sky")])
    keep = rm == 0
    dis = np.maximum.reduce([np.abs(rend[i] - rend[j]).mean(-1) for i in range(3) for j in range(i + 1, 3)])
    best, box = -1, (W // 2 - CW // 2, H // 2 - CH // 2)
    for y0 in range(0, H - CH + 1, 20):
        for x0 in range(0, W - CW + 1, 20):
            if keep[y0:y0 + CH, x0:x0 + CW].mean() < 0.999:
                continue
            m = dis[y0:y0 + CH, x0:x0 + CW].mean()
            if m > best:
                best, box = m, (x0, y0)
    if rm.any():  # ridge crop centred on the mask boundary at the median boundary column
        cols_ = np.where(rm.any(0))[0]
        x = int(np.median(cols_))
        y = int(np.where(rm[:, x] > 0)[0].max()) if rm[:, x].any() else H // 3
        ridge = (int(np.clip(x - CW // 2, 0, W - CW)), int(np.clip(y - CH // 2, 0, H - CH)))
    else:
        ridge = None
    return ridge, box


def zoom_panel(s, c):
    ridge, det = crops_for(s)
    src = photo_with_edges(s) if c == 0 else img(R / "runs" / V[c - 1] / "renders" / "test" / f"{s}.png")
    out = Image.new("RGB", (PW, PH), (255, 255, 255))
    boxes = [b for b in (ridge, det) if b is not None]
    for i, (x0, y0) in enumerate(boxes):
        crop = Image.fromarray(src[y0:y0 + CH, x0:x0 + CW]).resize((PW if len(boxes) == 1 else PW // 2, PH), Image.LANCZOS)
        out.paste(crop, (i * (PW // 2) if len(boxes) == 2 else 0, 0))
    lab = ("ridge | detail" if len(boxes) == 2 else "detail") + f"   ({s})"
    return caption(out, lab)


summary(zoom_panel, "summary_zoom.png", "  - zoom")


# ---- largest effects (selected BY effect size, labelled as such) ----
delta = sorted((f(s, "agent_keep_psnr") - f(s, "nomask_keep_psnr"), s) for s in rows)
pick = [s for _, s in delta[::-1][:3]] + [s for _, s in delta[:2]]
print("extremes (agent - nomask keep PSNR):", [(s, round(d, 2)) for d, s in delta[::-1][:3] + delta[:2]])
summary(full_panel, "extremes_masking_effect.png", "")
