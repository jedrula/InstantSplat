#!/usr/bin/env python3
"""
DINOv2 feature distillation into 3DGS Gaussians.

Loads a trained PLY, adds k per-Gaussian DINO feature attributes, trains them
via gsplat differentiable rendering against PCA-compressed DINOv2 ground truth.
Geometry is frozen — only per-Gaussian features are optimised.

Runtime: ~5 min for 3000 iters on 428K Gaussians.

Usage:
  PYTHONPATH=/workdir/gsplat python distill.py \
    --ply point_cloud.ply \
    --dino-dir /path/to/dino_output/ \
    --scene-dir /path/to/scene/ \
    --output-ply dino_gaussians.ply \
    [--iters 3000]
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import trange

GSPLAT_REPO = "/home/communications/workdir/gsplat"


def ensure_gsplat():
    if GSPLAT_REPO not in sys.path:
        sys.path.insert(0, GSPLAT_REPO)
    from gsplat import rasterization  # noqa


def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    props = [p.name for p in v.properties]

    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    opacity_raw = v["opacity"].astype(np.float32)
    scale_names = sorted([p for p in props if p.startswith("scale_")],
                         key=lambda s: int(s.split("_")[1]))
    scales_raw = np.stack([v[s] for s in scale_names], 1).astype(np.float32)
    rot_names = sorted([p for p in props if p.startswith("rot_")],
                       key=lambda s: int(s.split("_")[1]))
    quats = np.stack([v[r] for r in rot_names], 1).astype(np.float32)
    return {"xyz": xyz, "opacity_raw": opacity_raw, "scales_raw": scales_raw, "quats": quats}


def save_dino_ply(ply_path_in, dino_feats_np, output_path):
    """Append dino_0..dino_{k-1} attributes to existing PLY."""
    plydata = PlyData.read(ply_path_in)
    v = plydata["vertex"]
    n = len(v.data)
    k = dino_feats_np.shape[1]
    assert dino_feats_np.shape == (n, k)

    old_dtypes = [(p.name, v.data.dtype[p.name]) for p in v.properties]
    new_dtypes = old_dtypes + [(f"dino_{i}", "f4") for i in range(k)]
    new_data = np.empty(n, dtype=new_dtypes)
    for p in v.properties:
        new_data[p.name] = v.data[p.name]
    for i in range(k):
        new_data[f"dino_{i}"] = dino_feats_np[:, i]

    PlyData([PlyElement.describe(new_data, "vertex")], text=False).write(output_path)
    print(f"  Saved DINO PLY ({n:,} Gaussians, {k}D features): {output_path}")


# ── Camera loaders (same as langsplat, reused) ───────────────────────────────

def _qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),   2*(x*z+y*w)  ],
        [2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x**2+y**2)],
    ])


def load_cameras(scene_dir, dino_dir):
    """Match cameras to their dino_comp feature maps. Returns list of dicts."""
    transforms_path = os.path.join(scene_dir, "transforms.json")
    if os.path.exists(transforms_path):
        return _load_ns(transforms_path, dino_dir)
    sparse = os.path.join(scene_dir, "sparse", "0")
    if not os.path.exists(sparse):
        sparse = os.path.join(scene_dir, "sparse")
    return _load_colmap(scene_dir, sparse, dino_dir)


def _load_ns(transforms_path, dino_dir):
    with open(transforms_path) as f:
        data = json.load(f)
    cams = []
    for frame in data["frames"]:
        img_path = frame["file_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(transforms_path), img_path)
        for ext in ["", ".jpg", ".jpeg", ".png", ".webp"]:
            if os.path.exists(img_path + ext):
                img_path = img_path + ext; break
        if not os.path.exists(img_path):
            continue
        stem = Path(img_path).stem
        comp = Path(dino_dir) / f"{stem}_dino_comp.npy"
        if not comp.exists():
            continue
        fl_x = frame.get("fl_x") or data.get("fl_x")
        fl_y = frame.get("fl_y") or data.get("fl_y") or fl_x
        cx   = frame.get("cx")   or data.get("cx")
        cy   = frame.get("cy")   or data.get("cy")
        W    = int(frame.get("w") or data.get("w"))
        H    = int(frame.get("h") or data.get("h"))
        K = np.array([[fl_x,0,cx],[0,fl_y,cy],[0,0,1]], dtype=np.float32)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w).astype(np.float32)
        cams.append({"K": K, "w2c": w2c, "W": W, "H": H, "comp": str(comp)})
    return cams


def _load_colmap(scene_dir, sparse_dir, dino_dir):
    try:
        import pycolmap
        recon = pycolmap.Reconstruction(sparse_dir)
        cams = []
        for _, image in recon.images.items():
            stem = Path(image.name).stem
            comp = Path(dino_dir) / f"{stem}_dino_comp.npy"
            if not comp.exists():
                continue
            img_path = os.path.join(scene_dir, "images", image.name)
            if not os.path.exists(img_path):
                continue
            cam = image.camera
            K = np.array([[cam.focal_length_x, 0, cam.principal_point_x],
                          [0, cam.focal_length_y, cam.principal_point_y],
                          [0, 0, 1]], dtype=np.float32)
            cfw = image.cam_from_world()
            R = cfw.rotation.matrix()
            t = cfw.translation.reshape(3, 1)
            w2c = np.vstack([np.hstack([R, t]), [0,0,0,1]]).astype(np.float32)
            cams.append({"K": K, "w2c": w2c, "W": cam.width, "H": cam.height, "comp": str(comp)})
        return cams
    except Exception:
        # Text fallback
        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_txt  = os.path.join(sparse_dir, "images.txt")
        cam_data = {}
        with open(cameras_txt) as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                p = line.split()
                cid, model, W, H = int(p[0]), p[1], int(p[2]), int(p[3])
                if model == "PINHOLE":
                    fx, fy, cx, cy = float(p[4]), float(p[5]), float(p[6]), float(p[7])
                else:
                    f_ = float(p[4]); fx = fy = f_; cx, cy = float(p[5]), float(p[6])
                cam_data[cid] = (W, H, fx, fy, cx, cy)
        cams = []
        with open(images_txt) as f:
            lines = [l for l in f if not l.startswith("#") and l.strip()]
        for i in range(0, len(lines), 2):
            p = lines[i].split()
            cid, name = int(p[8]), p[9]
            stem = Path(name).stem
            comp = Path(dino_dir) / f"{stem}_dino_comp.npy"
            if not comp.exists(): continue
            img_path = os.path.join(scene_dir, "images", name)
            if not os.path.exists(img_path): continue
            W, H, fx, fy, cx, cy = cam_data[cid]
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
            q = [float(x) for x in p[1:5]]; t = np.array([float(x) for x in p[5:8]])
            R = _qvec2rot(q)
            w2c = np.vstack([np.hstack([R, t.reshape(3,1)]), [0,0,0,1]]).astype(np.float32)
            cams.append({"K": K, "w2c": w2c, "W": W, "H": H, "comp": str(comp)})
        return cams


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    ensure_gsplat()
    from gsplat import rasterization
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DINO distill] device={device}  iters={args.iters}")

    gs = load_ply(args.ply)
    N = gs["xyz"].shape[0]
    print(f"[DINO distill] {N:,} Gaussians")

    means   = torch.from_numpy(gs["xyz"]).to(device)
    quats   = F.normalize(torch.from_numpy(gs["quats"]).to(device), dim=-1)
    scales  = torch.exp(torch.from_numpy(gs["scales_raw"]).to(device))
    opacities = torch.sigmoid(torch.from_numpy(gs["opacity_raw"]).to(device))

    # Load PCA metadata to get n_components
    meta_path = os.path.join(args.dino_dir, "dino_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    k = meta["n_components"]
    print(f"[DINO distill] Feature dim: {k}D")

    dino_feats = torch.zeros(N, k, device=device, requires_grad=True)
    optimizer  = torch.optim.Adam([dino_feats], lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)

    cameras = load_cameras(args.scene_dir, args.dino_dir)
    if not cameras:
        print("ERROR: no cameras matched to DINO feature maps."); sys.exit(1)
    print(f"[DINO distill] {len(cameras)} cameras matched")

    # Render at DINO feature map resolution (much smaller than full image)
    # Scale intrinsics accordingly
    sample = np.load(cameras[0]["comp"])   # (H_p, W_p, k)
    feat_h, feat_w = sample.shape[:2]
    print(f"[DINO distill] Render resolution: {feat_w}×{feat_h}")

    import time
    t0 = time.time()
    pbar = trange(args.iters, desc="distilling")
    for step in pbar:
        cam = cameras[step % len(cameras)]

        # Scale K from full-image to dino-patch resolution
        sx = feat_w / cam["W"]
        sy = feat_h / cam["H"]
        K = cam["K"].copy()
        K[0] *= sx; K[1] *= sy
        Kt = torch.from_numpy(K).unsqueeze(0).to(device)
        w2ct = torch.from_numpy(cam["w2c"]).unsqueeze(0).to(device)

        # Load GT feature map (already at patch resolution)
        gt_np = np.load(cam["comp"]).astype(np.float32)  # (H_p, W_p, k)
        gt = torch.from_numpy(gt_np).to(device)

        renders, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=dino_feats,
            viewmats=w2ct, Ks=Kt,
            width=feat_w, height=feat_h,
            sh_degree=None, near_plane=0.01, far_plane=1000.0,
        )
        loss = F.mse_loss(renders[0], gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0 or step == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            pbar.set_postfix_str(f"loss={loss.item():.5f}  {rate:.1f}it/s")

    elapsed = time.time() - t0
    print(f"[DINO distill] Training: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    dino_np = dino_feats.detach().cpu().float().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ply)), exist_ok=True)
    save_dino_ply(args.ply, dino_np, args.output_ply)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",        required=True)
    parser.add_argument("--dino-dir",   required=True, help="Dir with dino_meta.json + *_dino_comp.npy")
    parser.add_argument("--scene-dir",  required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--iters",  type=int,   default=3000)
    parser.add_argument("--lr",     type=float, default=5e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
