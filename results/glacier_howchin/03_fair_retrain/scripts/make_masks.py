"""Build full-resolution (4000x3000) REMOVE masks (uint8, 255 = blacked out) for the 583
registered frames of the shared COLMAP model.

  agent    : the exact sam3_agent masks behind the original 'Ours' row. Keep-mask recovered from
             masked/images/*.png (2048x1536, pure black = removed), upsampled bilinear + >0.5
             threshold, i.e. the same full-res path as sam3_agent's pipeline (output_full_resolution).
  sam3_sky : 'SAM3, no agent' = sam3_prompt's own mask builder with its defaults
             (prompt "sky", score_thr 0.5, mask_threshold 0.5, min_area 64, dilate 15) run on the
             original 4000x3000 JPG. No detection -> empty mask (frame kept whole), matching the
             batch tool which skips such frames.

usage: make_masks.py agent            (sam3 env, CPU)
       make_masks.py sam3_sky         (sam3 env, GPU)
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE = Path("/mnt/windows/Dataset/2016-11-28_Howchin-AlphLake_Imagery-Files.beh")
R = BASE / "retrain_fair"
W, H = 4000, 3000


def names():
    from_txt = R / "shared_sparse_txt" / "images.txt"
    out = []
    lines = [l for l in open(from_txt) if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) >= 10:
            out.append(Path(p[9]).stem)
    return sorted(out)


def agent():
    out_dir = R / "masks" / "agent"
    stats = []
    for s in names():
        src = np.asarray(Image.open(BASE / "masked" / "images" / f"{s}.png").convert("RGB"))
        keep = (src.max(-1) > 0).astype(np.float32)
        keep_full = cv2.resize(keep, (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        remove = (~keep_full).astype(np.uint8) * 255
        Image.fromarray(remove, mode="L").save(out_dir / f"{s}.png", optimize=True)
        stats.append(dict(stem=s, removed_frac=float((remove > 0).mean())))
    json.dump(stats, open(out_dir / "_stats.json", "w"), indent=0)
    fr = np.array([x["removed_frac"] for x in stats])
    print(f"agent: {len(stats)} masks, frames with removal {int((fr > 0).sum())}, mean removed {fr.mean():.4f}")


def sam3_sky():
    sys.path.insert(0, "/home/otter77/git_project/sam3_prompt/src")
    from sam3_prompt.pipeline import RemovalConfig, build_mask
    from sam3_prompt.segment import Sam3Segmenter

    cfg = RemovalConfig()
    seg = Sam3Segmenter("facebook/sam3", device="cuda")
    out_dir = R / "masks" / "sam3_sky"
    stats = []
    t0 = time.time()
    for i, s in enumerate(names()):
        img = Image.open(BASE / "unmasked" / "images" / f"{s}.JPG").convert("RGB")
        res = seg.segment(img, "sky", score_thr=cfg.score_thr, mask_threshold=cfg.mask_threshold)
        try:
            m = build_mask(res, cfg)
            note = "ok"
        except RuntimeError as e:
            m = np.zeros((img.height, img.width), np.uint8)
            note = f"empty: {e}"
        assert m.shape == (H, W), m.shape
        Image.fromarray(m, mode="L").save(out_dir / f"{s}.png", optimize=True)
        stats.append(dict(stem=s, removed_frac=float((m > 0).mean()), n_masks=int(len(res.masks)),
                          scores=[round(float(x), 4) for x in res.scores.tolist()], note=note))
        if i % 50 == 0:
            print(f"  {i} {s} removed={stats[-1]['removed_frac']:.3f} {note[:40]} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(stats, open(out_dir / "_stats.json", "w"), indent=0)
    fr = np.array([x["removed_frac"] for x in stats])
    print(f"sam3_sky: {len(stats)} masks, frames with removal {int((fr > 0).sum())}, mean removed {fr.mean():.4f}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    {"agent": agent, "sam3_sky": sam3_sky}[sys.argv[1]]()
