#!/usr/bin/env python3
"""
LangSplat Phase 3: Train per-Gaussian language features.

Given a trained 3DGS PLY file and compressed (H, W, 3) feature maps,
adds 3 language attributes per Gaussian and trains them via differentiable
rendering (gsplat) to match the ground-truth compressed CLIP features.

Geometry is frozen; only language features are optimized.

Runs with instantsplat env:
  PYTHONPATH=/workdir/gsplat python train_lang_gaussians.py \
    --ply /path/to/point_cloud.ply \
    --lang-dir /path/to/lang3_maps/ \
    --scene-dir /path/to/scene/ \
    --output-ply /path/to/lang_gaussians.ply \
    --iters 3000
"""

import argparse
import json
import math
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
    from gsplat import rasterization  # noqa: F401 — validates import


def load_gaussians_from_ply(ply_path):
    """Load Gaussian attributes from a 3DGS PLY file."""
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    props = [prop.name for prop in vertex.properties]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)  # (N,3)

    # Opacity: activation = sigmoid(raw)
    opacity_raw = vertex["opacity"]  # (N,)

    # Scales: activation = exp(raw)
    scale_names = [p for p in props if p.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda s: int(s.split("_")[1]))
    scales_raw = np.stack([vertex[s] for s in scale_names], axis=1)  # (N,3)

    # Rotation quaternion: (w, x, y, z) stored as rot_0..rot_3
    rot_names = [p for p in props if p.startswith("rot_")]
    rot_names = sorted(rot_names, key=lambda s: int(s.split("_")[1]))
    quats = np.stack([vertex[r] for r in rot_names], axis=1)  # (N,4) w,x,y,z

    # Colours (SH DC) for reference rendering — not used in lang training but handy
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    return {
        "xyz": xyz.astype(np.float32),
        "opacity_raw": opacity_raw.astype(np.float32),
        "scales_raw": scales_raw.astype(np.float32),
        "quats": quats.astype(np.float32),
        "f_dc": f_dc.astype(np.float32),
    }


def save_lang_ply(ply_path_in, lang_feats_np, output_path):
    """
    Copy the input PLY and append lang_0, lang_1, lang_2 per-Gaussian attributes.
    lang_feats_np: (N, 3) float32
    """
    plydata = PlyData.read(ply_path_in)
    vertex = plydata["vertex"]
    n = len(vertex.data)
    assert lang_feats_np.shape == (n, 3), \
        f"Expected ({n}, 3), got {lang_feats_np.shape}"

    # Build new structured array with existing properties + lang features
    old_props = [prop.name for prop in vertex.properties]
    old_dtypes = [(prop.name, vertex.data.dtype[prop.name]) for prop in vertex.properties]
    new_dtypes = old_dtypes + [("lang_0", "f4"), ("lang_1", "f4"), ("lang_2", "f4")]

    new_data = np.empty(n, dtype=new_dtypes)
    for name in old_props:
        new_data[name] = vertex.data[name]
    new_data["lang_0"] = lang_feats_np[:, 0]
    new_data["lang_1"] = lang_feats_np[:, 1]
    new_data["lang_2"] = lang_feats_np[:, 2]

    new_el = PlyElement.describe(new_data, "vertex")
    PlyData([new_el], text=False).write(output_path)
    print(f"  Saved lang PLY: {output_path}  ({n} Gaussians)")


def load_scene_cameras(scene_dir, lang_dir):
    """
    Load camera poses + image paths, find matching lang3 feature maps.
    Returns list of dicts: {K, w2c, img_path, lang3_path, W, H}
    Supports nerfstudio transforms.json or COLMAP sparse/0/.
    """
    transforms_path = os.path.join(scene_dir, "transforms.json")
    if os.path.exists(transforms_path):
        return _load_ns_cameras(transforms_path, lang_dir)
    else:
        sparse_dir = os.path.join(scene_dir, "sparse", "0")
        if not os.path.exists(sparse_dir):
            sparse_dir = os.path.join(scene_dir, "sparse")
        return _load_colmap_cameras(scene_dir, sparse_dir, lang_dir)


def _load_ns_cameras(transforms_path, lang_dir):
    """Load from nerfstudio transforms.json."""
    with open(transforms_path) as f:
        data = json.load(f)

    cameras = []
    lang_dir = Path(lang_dir)

    # Global intrinsics (may be overridden per frame)
    g_fl_x = data.get("fl_x")
    g_fl_y = data.get("fl_y")
    g_cx = data.get("cx")
    g_cy = data.get("cy")
    g_w = data.get("w")
    g_h = data.get("h")

    for frame in data.get("frames", []):
        img_path = frame["file_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(transforms_path), img_path)
        if not os.path.exists(img_path):
            # try adding extension
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                if os.path.exists(img_path + ext):
                    img_path = img_path + ext
                    break
        if not os.path.exists(img_path):
            continue

        stem = Path(img_path).stem
        # Look for corresponding lang3 file
        lang3_path = lang_dir / f"{stem}_lang3.npy"
        if not lang3_path.exists():
            continue

        fl_x = frame.get("fl_x", g_fl_x)
        fl_y = frame.get("fl_y", g_fl_y)
        cx = frame.get("cx", g_cx)
        cy = frame.get("cy", g_cy)
        W = frame.get("w", g_w)
        H = frame.get("h", g_h)
        if any(v is None for v in [fl_x, fl_y, cx, cy, W, H]):
            continue

        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)

        # nerfstudio c2w is OpenGL convention (Y up, -Z forward)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)  # (4,4)
        # Flip Y and Z to get OpenCV convention (Y down, +Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)

        cameras.append({
            "K": K, "w2c": w2c, "img_path": img_path,
            "lang3_path": str(lang3_path), "W": int(W), "H": int(H),
        })

    return cameras


def _load_colmap_cameras(scene_dir, sparse_dir, lang_dir):
    """Load from COLMAP sparse reconstruction using pycolmap."""
    try:
        import pycolmap
    except ImportError:
        return _load_colmap_cameras_text(scene_dir, sparse_dir, lang_dir)

    recon = pycolmap.Reconstruction(sparse_dir)
    lang_dir = Path(lang_dir)
    cameras = []

    for img_id, image in recon.images.items():
        stem = Path(image.name).stem
        lang3_path = lang_dir / f"{stem}_lang3.npy"
        if not lang3_path.exists():
            continue

        img_path = os.path.join(scene_dir, "images", image.name)
        if not os.path.exists(img_path):
            continue

        cam = image.camera
        fx = cam.focal_length_x
        fy = cam.focal_length_y
        cx = cam.principal_point_x
        cy = cam.principal_point_y
        W, H = cam.width, cam.height
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # pycolmap ≥ 3.9 API: cam_from_world() → Rigid3d
        cfw = image.cam_from_world()
        R = cfw.rotation.matrix()
        t = cfw.translation.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([R, t], axis=1),
                               np.array([[0, 0, 0, 1]])], axis=0).astype(np.float32)

        cameras.append({
            "K": K, "w2c": w2c, "img_path": img_path,
            "lang3_path": str(lang3_path), "W": int(W), "H": int(H),
        })

    return cameras


def _load_colmap_cameras_text(scene_dir, sparse_dir, lang_dir):
    """Minimal COLMAP text parser (fallback when pycolmap unavailable)."""
    cameras_file = os.path.join(sparse_dir, "cameras.txt")
    images_file = os.path.join(sparse_dir, "images.txt")
    if not (os.path.exists(cameras_file) and os.path.exists(images_file)):
        raise FileNotFoundError(f"COLMAP text files not found in {sparse_dir}")

    cams = {}
    with open(cameras_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])
            if model in ("PINHOLE",):
                fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                f_ = float(parts[4])
                fx = fy = f_
                cx, cy = float(parts[5]), float(parts[6])
            else:
                fx = fy = float(parts[4])
                cx, cy = float(parts[6]), float(parts[7])
            cams[cam_id] = {"W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

    def qvec2rot(q):
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)],
        ])

    lang_dir = Path(lang_dir)
    cameras = []
    with open(images_file) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        cam_id = int(parts[8])
        img_name = parts[9]
        qvec = [float(x) for x in parts[1:5]]
        tvec = np.array([float(x) for x in parts[5:8]])
        R = qvec2rot(qvec)
        t = tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([R, t], axis=1),
                               np.array([[0, 0, 0, 1]])], axis=0).astype(np.float32)
        c = cams[cam_id]
        K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], dtype=np.float32)
        img_path = os.path.join(scene_dir, "images", img_name)
        stem = Path(img_name).stem
        lang3_path = lang_dir / f"{stem}_lang3.npy"
        if not (os.path.exists(img_path) and lang3_path.exists()):
            continue
        cameras.append({
            "K": K, "w2c": w2c, "img_path": img_path,
            "lang3_path": str(lang3_path), "W": c["W"], "H": c["H"],
        })

    return cameras


def train(args):
    ensure_gsplat()
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LangSplat train] device={device}  iters={args.iters}")

    # Load Gaussian geometry (frozen)
    print(f"[LangSplat train] Loading PLY: {args.ply}")
    gs = load_gaussians_from_ply(args.ply)
    N = gs["xyz"].shape[0]
    print(f"[LangSplat train] {N:,} Gaussians")

    means  = torch.from_numpy(gs["xyz"]).to(device)           # (N,3)
    quats  = torch.from_numpy(gs["quats"]).to(device)         # (N,4)
    # Normalize quaternions
    quats  = F.normalize(quats, dim=-1)
    scales = torch.exp(torch.from_numpy(gs["scales_raw"]).to(device))   # (N,3)
    opacities = torch.sigmoid(torch.from_numpy(gs["opacity_raw"]).to(device))  # (N,)

    # Language features: (N, latent_dim), initialized small random
    latent_dim = args.latent_dim
    lang_feats = torch.zeros(N, latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([lang_feats], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)

    # Load camera data
    print(f"[LangSplat train] Loading scene cameras from {args.scene_dir}")
    cameras = load_scene_cameras(args.scene_dir, args.lang_dir)
    if not cameras:
        print("ERROR: no cameras with matching lang3 feature maps found.")
        sys.exit(1)
    print(f"[LangSplat train] {len(cameras)} cameras matched")

    # Training loop
    print(f"[LangSplat train] Training {args.iters} iters...")
    for step in trange(args.iters):
        cam = cameras[step % len(cameras)]
        W, H = cam["W"], cam["H"]

        viewmat = torch.from_numpy(cam["w2c"]).unsqueeze(0).to(device)   # (1,4,4)
        K_mat   = torch.from_numpy(cam["K"]).unsqueeze(0).to(device)      # (1,3,3)

        # Load GT lang3 feature map and resize to camera WxH
        gt_np = np.load(cam["lang3_path"]).astype(np.float32)  # (H//4, W//4, 3)
        gt_h, gt_w = gt_np.shape[:2]
        if gt_h != H or gt_w != W:
            import cv2
            gt_np = cv2.resize(gt_np, (W, H), interpolation=cv2.INTER_LINEAR)
        gt = torch.from_numpy(gt_np).to(device)  # (H, W, 3)

        # Render language features through alpha compositing
        # lang_feats as "colors" — shape (N, latent_dim)
        renders, alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=lang_feats,
            viewmats=viewmat,
            Ks=K_mat,
            width=W,
            height=H,
            sh_degree=None,
            near_plane=0.01,
            far_plane=1000.0,
        )
        rendered = renders[0]  # (H, W, latent_dim)

        loss = F.mse_loss(rendered, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0 or step == 0:
            trange.write(f"  step {step+1:5d}  loss={loss.item():.6f}")

    # Save output PLY with lang features
    lang_np = lang_feats.detach().cpu().float().numpy()  # (N, 3)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_ply)), exist_ok=True)
    save_lang_ply(args.ply, lang_np, args.output_ply)
    print(f"[LangSplat train] Done. Lang PLY: {args.output_ply}")


def main():
    parser = argparse.ArgumentParser(description="LangSplat language Gaussian training")
    parser.add_argument("--ply", required=True, help="Trained 3DGS PLY file")
    parser.add_argument("--lang-dir", required=True, help="Dir with *_lang3.npy compressed feature maps")
    parser.add_argument("--scene-dir", required=True, help="Scene dir (with transforms.json or sparse/)")
    parser.add_argument("--output-ply", required=True, help="Output PLY path")
    parser.add_argument("--iters", type=int, default=3000, help="Training iterations")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for lang features")
    parser.add_argument("--latent-dim", type=int, default=3, help="Latent dimension (must match AE)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
