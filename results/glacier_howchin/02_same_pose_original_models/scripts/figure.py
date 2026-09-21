"""Same-pose render comparison: No mask vs SAM3 (no agent) vs Agent, on the glacier scene.

Pose selection is rule-based (no hand picking):
  * candidate pairs (A, N): A = agent-held-out test frame, N = A+1 = no-agent-held-out test frame
    (the two test splits interleave as adjacent frames), agent mask non-trivial at A or N.
  * pairs ranked by fair held-out delta = agent keep-PSNR at A - SAM3 keep-PSNR at N;
    the 25th / 50th / 75th percentile pairs are shown.
For each pair three poses are rendered with every model, each model using its OWN COLMAP frame:
  photo pose A (novel for agent, trained-on for no-agent), photo pose N (the reverse), and the
  midpoint pose M between A and N (equally novel for all models: each saw exactly one endpoint).
"""
import csv
import json
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from gsrender import B, S, read_colmap_txt, intrinsics, load_ply, render, to_u8

OUT = f"{S}/out"
os.makedirs(OUT, exist_ok=True)
W, H = 1000, 750
VARIANTS = ["unmasked", "native_prompt", "masked"]
LABEL = {"unmasked": "No mask (raw imagery)", "native_prompt": "SAM3, no agent", "masked": "Ours (agent + validation)"}

rows = {r["stem"]: r for r in csv.DictReader(open(f"{S}/sweep.csv"))}
num = lambda r, k: float(r[k])
stem = lambda n: n.rsplit(".", 1)[0]
test = {v: set(map(stem, json.load(open(f"{B}/{v}/runs/my_experiment/split.json"))["test"])) for v in VARIANTS}

pairs = []
for a in sorted(test["masked"]):
    n = f"DJI_{int(a[4:]) + 1:04d}"
    if a in rows and n in rows and n in test["unmasked"] and n in test["native_prompt"]:
        ra, rn = rows[a], rows[n]
        if min(num(ra, "keep_frac"), num(rn, "keep_frac")) < 0.95:
            delta = num(ra, "masked_psnr_keep") - num(rn, "native_prompt_psnr_keep")
            pairs.append((delta, a, n))
pairs.sort()
print("candidate pairs:", len(pairs))
picks = [pairs[int(round(q * (len(pairs) - 1)))] for q in (0.25, 0.5, 0.75)]
print("picked (delta, A, N):", [(round(d, 2), a, n) for d, a, n in picks])


def quat_from_R(R):
    m = R
    w = np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2
    x = np.copysign(x, m[2, 1] - m[1, 2])
    y = np.copysign(y, m[0, 2] - m[2, 0])
    z = np.copysign(z, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def R_from_quat(q):
    from gsrender import qvec2rotmat
    return qvec2rotmat(q / np.linalg.norm(q))


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def mid_pose(ia, ib):
    Ca, Cb = -ia["R"].T @ ia["t"], -ib["R"].T @ ib["t"]
    Rm = R_from_quat(slerp(quat_from_R(ia["R"]), quat_from_R(ib["R"]), 0.5))
    Cm = (Ca + Cb) / 2
    return Rm, -Rm @ Cm


poses = {v: read_colmap_txt(v) for v in VARIANTS}
views = []  # (key, pose_kind, stem_for_photo)
for _, a, n in picks:
    views += [(f"{a}", "A", a), (f"mid_{a}_{n}", "M", None), (f"{n}", "N", n)]

renders = {}
for v in VARIANTS:
    cams, imgs = poses[v]
    model = load_ply(f"{B}/{v}/runs/my_experiment/checkpoints/point_cloud_30000.ply")
    for _, a, n in picks:
        ia, ib = imgs[a], imgs[n]
        K = intrinsics(cams[ia["cam"]], W, H)
        renders[(v, a)] = to_u8(render(model, ia["R"], ia["t"], K, W, H))
        renders[(v, n)] = to_u8(render(model, ib["R"], ib["t"], K, W, H))
        Rm, tm = mid_pose(ia, ib)
        renders[(v, f"mid_{a}_{n}")] = to_u8(render(model, Rm, tm, K, W, H))
        # camera baseline between the two frames in this model's units vs scene scale sanity
        Ca, Cb = -ia["R"].T @ ia["t"], -ib["R"].T @ ib["t"]
        print(f"  {v:13s} {a}->{n} baseline {np.linalg.norm(Ca - Cb):.4f}", flush=True)
    del model
    torch.cuda.empty_cache()

for key, img in renders.items():
    Image.fromarray(img).save(f"{OUT}/{key[1]}__{key[0]}.png")


def photo(s):
    im = Image.open(f"{B}/unmasked/images/{s}.JPG")
    im.draft("RGB", (W * 2, H * 2))
    return np.asarray(im.convert("RGB").resize((W, H), Image.LANCZOS))


def keep_mask(s):
    m = np.asarray(Image.open(f"{B}/masked/images/{s}.png").convert("RGB")).max(-1) > 0
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127


def outline(img, keep):
    out = img.copy()
    e = keep ^ np.roll(keep, 1, 0) | keep ^ np.roll(keep, 1, 1)
    e = e | np.roll(e, 1, 0) | np.roll(e, 1, 1) | np.roll(e, -1, 0) | np.roll(e, -1, 1)
    out[e] = (255, 40, 40)
    return out


try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
except OSError:
    FONT = FONT_S = ImageFont.load_default()

PW, PH = 500, 375  # panel size in the grid
cols = ["photo"] + VARIANTS
head_h, row_lab_w = 40, 0


def panel(arr, caption):
    im = Image.fromarray(arr).resize((PW, PH), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle([0, PH - 28, PW, PH], fill=(0, 0, 0))
    d.text((8, PH - 25), caption, fill=(255, 255, 255), font=FONT_S)
    return im


def split_tag(v, s):
    return "held-out" if s in test[v] else "trained-on"


for pi, (delta, a, n) in enumerate(picks):
    grid = Image.new("RGB", (PW * len(cols), head_h + PH * 3), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    for ci, c in enumerate(["Real photo (red = agent mask edge)"] + [LABEL[v] for v in VARIANTS]):
        d.text((ci * PW + 8, 9), c, fill=(0, 0, 0), font=FONT)
    for ri, (kind, s) in enumerate([("A", a), ("M", None), ("N", n)]):
        y = head_h + ri * PH
        key = s if s else f"mid_{a}_{n}"
        if s:
            ph = outline(photo(s), keep_mask(s))
            grid.paste(panel(ph, f"{s} (camera pose of this photo)"), (0, y))
        else:
            blank = np.full((H, W, 3), 235, np.uint8)
            im = panel(blank, "no photo at this pose")
            dd = ImageDraw.Draw(im)
            dd.text((20, 120), f"Midpoint pose between\n{a} and {n}\n\nequally unseen by all models\n(each trained on exactly one\nof the two endpoint frames)", fill=(60, 60, 60), font=FONT)
            grid.paste(im, (0, y))
        for ci, v in enumerate(VARIANTS, start=1):
            tag = split_tag(v, s) if s else "novel for all"
            grid.paste(panel(renders[(v, key)], f"render @ {'midpoint' if not s else s} [{tag}]"), (ci * PW, y))
    path = f"{OUT}/pose_compare_{pi + 1}_{a}_{n}.png"
    grid.save(path)
    print("wrote", path)

# per-pair numbers for the report
with open(f"{OUT}/picked_pairs.json", "w") as f:
    json.dump([dict(A=a, N=n, fair_heldout_delta_keep_psnr=d,
                    A_keep_frac=num(rows[a], "keep_frac"), N_keep_frac=num(rows[n], "keep_frac"),
                    scores_A={v: dict(split=rows[a][f"{v}_split"], psnr_keep=num(rows[a], f"{v}_psnr_keep"), psnr_full=num(rows[a], f"{v}_psnr_full")) for v in VARIANTS},
                    scores_N={v: dict(split=rows[n][f"{v}_split"], psnr_keep=num(rows[n], f"{v}_psnr_keep"), psnr_full=num(rows[n], f"{v}_psnr_full")) for v in VARIANTS})
               for d, a, n in picks], f, indent=1)
print("pairs:", len(pairs), "delta quantiles", np.round(np.quantile([p[0] for p in pairs], [0, .25, .5, .75, 1]), 2).tolist())
