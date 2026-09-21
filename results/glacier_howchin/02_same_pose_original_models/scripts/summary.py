"""Compact agent vs no-agent figure at the fair midpoint poses, with two auto-selected zoom crops:
  * ridge crop  : centred on the agent's sky/terrain boundary (where masking acts)
  * detail crop : window with the largest |agent - SAM3| difference fully inside terrain the agent kept
Uses renders saved by figure.py (no GPU)."""
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = os.path.dirname(os.path.abspath(__file__))
OUT = f"{S}/out"
picks = json.load(open(f"{OUT}/picked_pairs.json"))
CW, CH = 250, 188  # crop size in 1000x750 render pixels
PW, PH = 480, 360

F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)


def load(key, v):
    return np.asarray(Image.open(f"{OUT}/{key}__{v}.png")).astype(np.int16)


def ridge_box(agent):
    kept = agent.max(-1) > 6
    cols = np.arange(agent.shape[1])
    top = np.array([np.argmax(kept[:, x]) if kept[:, x].any() else -1 for x in cols])
    valid = top > 0  # columns where something above was blacked out
    if not valid.any():
        return None
    xs = cols[valid]
    x = int(np.median(xs))
    y = int(top[x]) if top[x] > 0 else int(np.median(top[valid]))
    x0 = int(np.clip(x - CW // 2, 0, agent.shape[1] - CW))
    y0 = int(np.clip(y - CH // 2, 0, agent.shape[0] - CH))
    return x0, y0


def detail_box(agent, other):
    kept = agent.max(-1) > 6
    diff = np.abs(agent - other).mean(-1)
    best, box = -1, None
    for y0 in range(0, agent.shape[0] - CH + 1, 20):
        for x0 in range(0, agent.shape[1] - CW + 1, 20):
            if kept[y0:y0 + CH, x0:x0 + CW].mean() < 0.999:
                continue
            m = diff[y0:y0 + CH, x0:x0 + CW].mean()
            if m > best:
                best, box = m, (x0, y0)
    return box


def panel(arr, caption, box=None, color=None):
    im = Image.fromarray(arr.astype(np.uint8)).resize((PW, PH), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    if box is not None:
        sx, sy = PW / arr.shape[1], PH / arr.shape[0]
        for (x0, y0), c in box:
            d.rectangle([x0 * sx, y0 * sy, (x0 + CW) * sx, (y0 + CH) * sy], outline=c, width=3)
    d.rectangle([0, PH - 26, PW, PH], fill=(0, 0, 0))
    d.text((8, PH - 23), caption, fill=(255, 255, 255), font=FS)
    return im


def crop(arr, xy):
    x0, y0 = xy
    return arr[y0:y0 + CH, x0:x0 + CW]


headers = ["Without agent (SAM3)", "With agent (ours)", "Ridge zoom: without", "Ridge zoom: with",
           "Detail zoom: without", "Detail zoom: with"]
HEAD = 36
grid = Image.new("RGB", (PW * 6, HEAD + PH * len(picks)), (255, 255, 255))
d = ImageDraw.Draw(grid)
for i, h in enumerate(headers):
    d.text((i * PW + 8, 8), h, fill=(0, 0, 0), font=F)

YEL, CYA = (255, 210, 0), (0, 220, 255)
for r, p in enumerate(picks):
    key = f"mid_{p['A']}_{p['N']}"
    ag, na = load(key, "masked"), load(key, "native_prompt")
    rb, db = ridge_box(ag), detail_box(ag, na)
    boxes = [(b, c) for b, c in ((rb, YEL), (db, CYA)) if b is not None]
    y = HEAD + r * PH
    lab = f"midpoint {p['A'][4:]}/{p['N'][4:]}"
    grid.paste(panel(na, f"{lab} [novel view]", boxes), (0, y))
    grid.paste(panel(ag, f"{lab} [novel view]", boxes), (PW, y))
    for c, (b, name) in enumerate(((rb, "ridge"), (db, "detail"))):
        if b is None:
            continue
        grid.paste(panel(crop(na, b), f"{name} crop, 1.9x"), ((2 + 2 * c) * PW, y))
        grid.paste(panel(crop(ag, b), f"{name} crop, 1.9x"), ((3 + 2 * c) * PW, y))
    print(key, "ridge", rb, "detail", db)

path = f"{OUT}/summary_agent_vs_noagent_midpoint.png"
grid.save(path)
print("wrote", path, grid.size)
