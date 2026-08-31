#!/usr/bin/env python3
"""
Query a DINO-augmented 3DGS PLY.

Two modes:
  --mask PNG    : binary mask of a region in the rendered view → find matching Gaussians
  --query-image : a reference image crop (e.g. a hold photo) → feature similarity

Outputs a heatmap PNG overlaid on the rendered colour view.

Usage:
  PYTHONPATH=/workdir/gsplat python query.py \
    --dino-ply dino_gaussians.ply \
    --pca /path/to/dino_output/pca.npz \
    --scene-dir /path/to/scene/ \
    [--mask mask.png | --query-image hold.jpg] \
    --output heatmap.png
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
from plyfile import PlyData

GSPLAT_REPO = "/home/communications/workdir/gsplat"
DINO_PATCH = 14


def ensure_gsplat():
    if GSPLAT_REPO not in sys.path:
        sys.path.insert(0, GSPLAT_REPO)


def load_dino_ply(ply_path):
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

    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)

    dino_names = sorted([p for p in props if p.startswith("dino_")],
                        key=lambda s: int(s.split("_")[1]))
    if not dino_names:
        print("ERROR: PLY has no dino_* attributes. Run distill.py first.")
        sys.exit(1)
    dino_feats = np.stack([v[d] for d in dino_names], 1).astype(np.float32)
    k = len(dino_names)

    return {
        "xyz": xyz, "opacity_raw": opacity_raw, "scales_raw": scales_raw,
        "quats": quats, "sh_dc": sh_dc, "dino_feats": dino_feats, "k": k,
    }


def load_first_camera(scene_dir):
    """Return (w2c 4×4, K 3×3, W, H) for the first training camera."""
    transforms = os.path.join(scene_dir, "transforms.json")
    if os.path.exists(transforms):
        with open(transforms) as f:
            data = json.load(f)
        frame = data["frames"][0]
        fl_x = frame.get("fl_x") or data["fl_x"]
        fl_y = frame.get("fl_y") or data.get("fl_y") or fl_x
        cx   = frame.get("cx")   or data["cx"]
        cy   = frame.get("cy")   or data["cy"]
        W, H = int(frame.get("w") or data["w"]), int(frame.get("h") or data["h"])
        K = np.array([[fl_x,0,cx],[0,fl_y,cy],[0,0,1]], dtype=np.float32)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, 1:3] *= -1
        return np.linalg.inv(c2w).astype(np.float32), K, W, H

    sparse = os.path.join(scene_dir, "sparse", "0")
    if not os.path.exists(sparse):
        sparse = os.path.join(scene_dir, "sparse")
    import pycolmap
    recon = pycolmap.Reconstruction(sparse)
    image = next(iter(recon.images.values()))
    cam = image.camera
    K = np.array([[cam.focal_length_x, 0, cam.principal_point_x],
                  [0, cam.focal_length_y, cam.principal_point_y],
                  [0, 0, 1]], dtype=np.float32)
    cfw = image.cam_from_world()
    R = cfw.rotation.matrix(); t = cfw.translation.reshape(3,1)
    w2c = np.vstack([np.hstack([R,t]), [0,0,0,1]]).astype(np.float32)
    return w2c, K, cam.width, cam.height


def encode_image_query(query_img_path, pca_path, device):
    """Extract DINOv2 features from a query image crop, project via PCA → k-D vector."""
    pca = np.load(pca_path)
    mean, V = pca["mean"], pca["V"]
    k = V.shape[1]

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           pretrained=True, verbose=False).to(device).eval()
    dmean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
    dstd  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)

    img = Image.open(query_img_path).convert("RGB")
    # Resize to nearest multiple of 14
    w, h = img.size
    pw = max(14, round(w/14)*14)
    ph = max(14, round(h/14)*14)
    img = img.resize((pw, ph), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).float().to(device) / 255.0
    t = ((t.permute(2,0,1) - dmean) / dstd).unsqueeze(0)

    with torch.no_grad():
        feats = model.forward_features(t)["x_norm_patchtokens"]  # (1, N, 384)
    feats = feats.squeeze(0).cpu().numpy()        # (N, 384)
    mean_feat = feats.mean(0)                     # (384,)
    compressed = (mean_feat - mean) @ V           # (k,)
    return compressed.astype(np.float32)          # (k,)


def sh_dc_to_rgb(f_dc):
    return np.clip(f_dc * 0.28209479177387814 + 0.5, 0, 1)


def render_both(gs, w2c, K, W, H, device, feat_render_size=None):
    """Render colour + DINO feature map. Returns (colour HWC, feat HWk)."""
    from gsplat import rasterization

    means   = torch.from_numpy(gs["xyz"]).to(device)
    quats   = F.normalize(torch.from_numpy(gs["quats"]).to(device), dim=-1)
    scales  = torch.exp(torch.from_numpy(gs["scales_raw"]).to(device))
    opacs   = torch.sigmoid(torch.from_numpy(gs["opacity_raw"]).to(device))
    dino    = torch.from_numpy(gs["dino_feats"]).to(device)
    rgb     = torch.from_numpy(sh_dc_to_rgb(gs["sh_dc"])).to(device)

    w2ct = torch.from_numpy(w2c).unsqueeze(0).to(device)
    Kt   = torch.from_numpy(K).unsqueeze(0).to(device)

    with torch.no_grad():
        color_render, _, _ = rasterization(
            means, quats, scales, opacs, rgb, w2ct, Kt, W, H,
            sh_degree=None, near_plane=0.01, far_plane=1000.0)

        # Render features at smaller resolution if requested
        rW, rH = (feat_render_size or (W, H))
        rKt = Kt.clone()
        rKt[0, 0] *= rW / W; rKt[0, 1] *= rH / H
        feat_render, _, _ = rasterization(
            means, quats, scales, opacs, dino, w2ct, rKt, rW, rH,
            sh_degree=None, near_plane=0.01, far_plane=1000.0)

    return color_render[0].cpu().numpy(), feat_render[0].cpu().numpy()


def make_heatmap(relevance, color_img, alpha=0.6):
    import matplotlib.cm as cm
    cmap = cm.get_cmap("turbo")
    heat = cmap(relevance)[..., :3]
    blended = (1 - alpha) * color_img + alpha * heat
    return (blended.clip(0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-ply",     required=True, help="DINO-augmented PLY")
    parser.add_argument("--pca",          required=True, help="pca.npz from extract_features.py")
    parser.add_argument("--scene-dir",    required=True, help="Scene directory for camera pose")
    parser.add_argument("--mask",         help="Binary mask PNG of query region (white=query)")
    parser.add_argument("--query-image",  help="Image crop to use as query (alternative to mask)")
    parser.add_argument("--output",       default="dino_query.png")
    parser.add_argument("--alpha",        type=float, default=0.6, help="Heatmap overlay opacity")
    args = parser.parse_args()

    if not args.mask and not args.query_image:
        print("ERROR: provide --mask or --query-image"); sys.exit(1)

    ensure_gsplat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DINO query] device={device}")

    print("[DINO query] Loading DINO PLY...")
    gs = load_dino_ply(args.dino_ply)
    k  = gs["k"]
    print(f"  {gs['xyz'].shape[0]:,} Gaussians  {k}D features")

    print("[DINO query] Loading camera...")
    w2c, K, W, H = load_first_camera(args.scene_dir)

    # Feature render resolution: same as dino patch grid
    pca_data = np.load(args.pca)
    meta_path = os.path.join(os.path.dirname(args.pca), "dino_meta.json")
    feat_w = feat_h = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            m = json.load(f)
        feat_w = m.get("patch_grid_w", W // 14)
        feat_h = m.get("patch_grid_h", H // 14)
    feat_size = (feat_w or W // 14, feat_h or H // 14)

    print(f"[DINO query] Rendering at {W}×{H} (colour) and {feat_size[0]}×{feat_size[1]} (features)...")
    color_np, feat_np = render_both(gs, w2c, K, W, H, device, feat_render_size=feat_size)
    # feat_np: (feat_h, feat_w, k)

    # ── Build query vector ─────────────────────────────────────────────────────
    if args.mask:
        mask = np.array(Image.open(args.mask).convert("L").resize(
            (feat_size[0], feat_size[1]), Image.NEAREST)) > 128
        if not mask.any():
            print("ERROR: mask is empty"); sys.exit(1)
        query_vec = feat_np[mask].mean(0)          # (k,)
        print(f"  Query from mask: {mask.sum()} pixels")
    else:
        query_vec = encode_image_query(args.query_image, args.pca, device)
        print(f"  Query from image: {args.query_image}")

    # ── Cosine similarity ──────────────────────────────────────────────────────
    feat_flat = feat_np.reshape(-1, k)
    feat_norm = feat_flat / (np.linalg.norm(feat_flat, axis=1, keepdims=True) + 1e-6)
    q_norm    = query_vec / (np.linalg.norm(query_vec) + 1e-6)
    sim = (feat_norm @ q_norm).reshape(feat_size[1], feat_size[0])   # (feat_h, feat_w)

    # Normalise to [0, 1]
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)

    # Upsample relevance to colour image resolution
    sim_up = np.array(Image.fromarray((sim * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)) / 255.0

    heatmap = make_heatmap(sim_up, color_np.clip(0, 1), alpha=args.alpha)
    Image.fromarray(heatmap).save(args.output)
    print(f"[DINO query] Saved: {args.output}")

    # Also save raw similarity as npy
    npy_out = args.output.replace(".png", "_sim.npy")
    np.save(npy_out, sim_up)


if __name__ == "__main__":
    main()
