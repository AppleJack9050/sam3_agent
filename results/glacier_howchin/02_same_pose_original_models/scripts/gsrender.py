"""Minimal renderer for gsplat_eval PLY checkpoints + COLMAP TXT poses (read-only on source data)."""
import math
import numpy as np
import torch
from gsplat import rasterization

B = "/mnt/windows/Dataset/2016-11-28_Howchin-AlphLake_Imagery-Files.beh"
S = "/tmp/claude-1000/-home-otter77-git-project-sam3-agent/882a9da0-78a0-4c68-a66b-1f173fe2b5a0/scratchpad/render_cmp"


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_colmap_txt(variant):
    d = f"{S}/txt_{variant}"
    cams = {}
    for line in open(f"{d}/cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cams[int(p[0])] = dict(model=p[1], W=int(p[2]), H=int(p[3]), params=list(map(float, p[4:])))
    imgs = {}
    with open(f"{d}/images.txt") as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        name = p[9]
        stem = name.rsplit(".", 1)[0]
        imgs[stem] = dict(R=qvec2rotmat(q), t=t, cam=int(p[8]), name=name)
    return cams, imgs


def intrinsics(cam, W, H):
    """Pinhole K for rendering at W x H (SIMPLE_RADIAL distortion ignored)."""
    f, cx, cy = cam["params"][:3]
    sx, sy = W / cam["W"], H / cam["H"]
    return np.array([[f * sx, 0, cx * sx], [0, f * sy, cy * sy], [0, 0, 1]], dtype=np.float32)


def load_ply(path, device="cuda"):
    with open(path, "rb") as fh:
        n = None
        props = []
        while True:
            line = fh.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        off = fh.tell()
    arr = np.fromfile(path, dtype=np.float32, offset=off, count=n * len(props)).reshape(n, len(props))
    ix = {p: i for i, p in enumerate(props)}
    col = lambda names: torch.from_numpy(np.ascontiguousarray(arr[:, [ix[k] for k in names]])).to(device)
    means = col(["x", "y", "z"])
    sh0 = col([f"f_dc_{i}" for i in range(3)])[:, None, :]
    nrest = sum(p.startswith("f_rest_") for p in props)
    shN = col([f"f_rest_{i}" for i in range(nrest)]).reshape(n, 3, nrest // 3).transpose(1, 2)
    colors = torch.cat([sh0, shN], 1).contiguous()
    opac = torch.sigmoid(col(["opacity"])[:, 0])
    scales = torch.exp(col(["scale_0", "scale_1", "scale_2"]))
    quats = col(["rot_0", "rot_1", "rot_2", "rot_3"])
    quats = quats / quats.norm(dim=-1, keepdim=True)
    del arr
    sh_degree = int(math.isqrt(colors.shape[1]) - 1)
    return dict(means=means, quats=quats, scales=scales, opacities=opac, colors=colors, sh_degree=sh_degree)


@torch.no_grad()
def render(model, R, t, K, W, H, mode="classic"):
    # background defaults to black, matching training on black-masked images
    vm = torch.eye(4, device="cuda")
    vm[:3, :3] = torch.from_numpy(R).float()
    vm[:3, 3] = torch.from_numpy(t).float()
    img, _, _ = rasterization(
        model["means"], model["quats"], model["scales"], model["opacities"], model["colors"],
        vm[None], torch.from_numpy(K).cuda()[None], W, H,
        sh_degree=model["sh_degree"], rasterize_mode=mode,
    )
    return img[0].clamp(0, 1).cpu().numpy()


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
