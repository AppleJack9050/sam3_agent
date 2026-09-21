"""Check the renderer reproduces the saved gsplat_eval test renders (same model, same pose)."""
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, __import__("os").path.dirname(__file__))
from gsrender import B, read_colmap_txt, intrinsics, load_ply, render

CASES = [("masked", "DJI_0001", 512, 384), ("masked", "DJI_0250", 512, 384), ("unmasked", "DJI_0002", 500, 375)]

for variant in dict.fromkeys(c[0] for c in CASES):
    cams, imgs = read_colmap_txt(variant)
    model = load_ply(f"{B}/{variant}/runs/my_experiment/checkpoints/point_cloud_30000.ply")
    print(variant, "N =", model["means"].shape[0], "sh", model["sh_degree"], "registered", len(imgs), flush=True)
    for v, stem, W, H in CASES:
        if v != variant:
            continue
        ref_path = f"{B}/{variant}/runs/my_experiment/renders/test/{stem}.png"
        try:
            ref = np.asarray(Image.open(ref_path)).astype(np.float32) / 255
        except FileNotFoundError:
            print("  no saved render for", stem)
            continue
        im = imgs[stem]
        K = intrinsics(cams[im["cam"]], W, H)
        for mode in ["classic", "antialiased"]:
            out = render(model, im["R"], im["t"], K, W, H, mode=mode)
            mse = float(((out - ref) ** 2).mean())
            print(f"  {stem} {mode:12s} PSNR(ours vs saved render) = {10*np.log10(1/max(mse,1e-12)):.2f} dB", flush=True)
    del model
    torch.cuda.empty_cache()
