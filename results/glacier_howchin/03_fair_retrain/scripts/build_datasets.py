"""Build the three training datasets with an IDENTICAL pixel pipeline:

  original DJI JPG (4000x3000) -> black out REMOVE mask at full res (none for nomask)
  -> PIL LANCZOS to 1000x750 (exactly what gsplat_eval's loader does for --downsample 4)
  -> lossless PNG named <stem>.png

All variants share one COLMAP model (shared_sparse/0, names rewritten .JPG -> .png), so they have
the same poses, the same init points, and the same every-8th test split (73 views).
Because the PNGs are already at the training size, gsplat_eval neither resizes for training nor for
evaluation, so training GT == evaluation GT for every variant.
"""
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path("/mnt/windows/Dataset/2016-11-28_Howchin-AlphLake_Imagery-Files.beh")
R = BASE / "retrain_fair"
SIZE = (1000, 750)
VARIANTS = {"nomask": None, "sam3_noagent": "sam3_sky", "agent": "agent"}


def stems():
    return sorted(p.stem for p in (R / "masks" / "agent").glob("DJI_*.png"))


def one(stem):
    full = np.asarray(Image.open(BASE / "unmasked" / "images" / f"{stem}.JPG").convert("RGB"))
    for v, mask_name in VARIANTS.items():
        arr = full
        if mask_name is not None:
            rm = np.asarray(Image.open(R / "masks" / mask_name / f"{stem}.png")) > 0
            assert rm.shape == full.shape[:2]
            arr = full.copy()
            arr[rm] = 0
        Image.fromarray(arr).resize(SIZE, Image.LANCZOS).save(R / "data" / v / "images" / f"{stem}.png")
    return stem


if __name__ == "__main__":
    ss = stems()
    assert len(ss) == 583, len(ss)
    assert len(list((R / "masks" / "sam3_sky").glob("DJI_*.png"))) == 583
    for v in VARIANTS:
        (R / "data" / v / "images").mkdir(parents=True, exist_ok=True)
        sp = R / "data" / v / "sparse" / "0"
        sp.mkdir(parents=True, exist_ok=True)
        for f in (R / "shared_sparse" / "0").iterdir():
            shutil.copy2(f, sp / f.name)
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count())) as ex:
        for i, s in enumerate(ex.map(one, ss, chunksize=4)):
            if i % 100 == 0:
                print("built", i, s, flush=True)
    for v in VARIANTS:
        print(v, len(list((R / "data" / v / "images").glob("*.png"))), "images")
