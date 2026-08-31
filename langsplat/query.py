#!/usr/bin/env python3
"""
LangSplat Phase 4: Query the language Gaussians with a text prompt.

Given a language-augmented PLY, an autoencoder checkpoint, and a text query,
renders a relevance heatmap from the specified camera viewpoint.

Runs with instantsplat env:
  PYTHONPATH=/workdir/gsplat python query.py \
    --lang-ply /path/to/lang_gaussians.ply \
    --autoencoder /path/to/autoencoder.pth \
    --query "red climbing hold" \
    --camera-json /path/to/initial_camera.json \
    --scene-dir /path/to/scene_dir \
    --output heatmap.png

Output: a PNG heatmap overlaid on the rendered color view.
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
import open_clip
from PIL import Image
from plyfile import PlyData

GSPLAT_REPO = "/home/communications/workdir/gsplat"


def ensure_gsplat():
    if GSPLAT_REPO not in sys.path:
        sys.path.insert(0, GSPLAT_REPO)
    from gsplat import rasterization  # noqa — validates


class LangAutoencoder(torch.nn.Module):
    def __init__(self, feat_dim=512, latent_dim=3):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 256), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 256), torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, feat_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_lang_ply(ply_path):
    """Load PLY with lang_0, lang_1, lang_2 language attributes."""
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    props = [p.name for p in vertex.properties]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    opacity_raw = vertex["opacity"].astype(np.float32)

    scale_names = sorted([p for p in props if p.startswith("scale_")],
                         key=lambda s: int(s.split("_")[1]))
    scales_raw = np.stack([vertex[s] for s in scale_names], axis=1).astype(np.float32)

    rot_names = sorted([p for p in props if p.startswith("rot_")],
                       key=lambda s: int(s.split("_")[1]))
    quats = np.stack([vertex[r] for r in rot_names], axis=1).astype(np.float32)

    # SH DC for colour rendering
    f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)

    has_lang = all(f"lang_{i}" in props for i in range(3))
    lang_feats = None
    if has_lang:
        lang_feats = np.stack([vertex["lang_0"], vertex["lang_1"], vertex["lang_2"]], axis=1).astype(np.float32)
    else:
        print("WARNING: PLY has no lang_0/1/2 attributes — did you run train_lang_gaussians.py?")

    return {
        "xyz": xyz, "opacity_raw": opacity_raw, "scales_raw": scales_raw,
        "quats": quats, "f_dc": f_dc, "lang_feats": lang_feats,
    }


def load_autoencoder(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feat_dim = ckpt["feat_dim"]
    latent_dim = ckpt["latent_dim"]
    model = LangAutoencoder(feat_dim=feat_dim, latent_dim=latent_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def encode_text_query(query: str, ae_model, device):
    """CLIP-encode text query, then pass through AE encoder → latent 3D vector."""
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tokens)  # (1, 512)
        text_feat = F.normalize(text_feat, dim=-1)  # L2 normalize
        text_z = ae_model.encode(text_feat)          # (1, 3)

    return text_z.squeeze(0)  # (3,)


def load_first_training_camera(scene_dir):
    """
    Load the first training camera pose+intrinsics from a scene directory.
    Supports nerfstudio transforms.json or COLMAP sparse/0/.
    Returns (w2c 4x4 float32, K 3x3 float32, W int, H int).
    """
    transforms_path = os.path.join(scene_dir, "transforms.json")
    if os.path.exists(transforms_path):
        with open(transforms_path) as f:
            data = json.load(f)
        frame = data["frames"][0]
        fl_x = frame.get("fl_x") or data.get("fl_x")
        fl_y = frame.get("fl_y") or data.get("fl_y") or fl_x
        cx   = frame.get("cx")   or data.get("cx")
        cy   = frame.get("cy")   or data.get("cy")
        W    = int(frame.get("w") or data.get("w"))
        H    = int(frame.get("h") or data.get("h"))
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[:3, 1:3] *= -1  # nerfstudio OpenGL → OpenCV
        w2c = np.linalg.inv(c2w).astype(np.float32)
        return w2c, K, W, H

    # Fall back to COLMAP
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    if not os.path.exists(sparse_dir):
        sparse_dir = os.path.join(scene_dir, "sparse")
    if os.path.exists(sparse_dir):
        try:
            import pycolmap
            recon = pycolmap.Reconstruction(sparse_dir)
            image = next(iter(recon.images.values()))
            cam = image.camera
            fx, fy = cam.focal_length_x, cam.focal_length_y
            cx, cy = cam.principal_point_x, cam.principal_point_y
            W, H = cam.width, cam.height
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            cfw = image.cam_from_world()
            R = cfw.rotation.matrix()
            t = cfw.translation.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([R, t], axis=1),
                                   np.array([[0, 0, 0, 1]])], axis=0).astype(np.float32)
            return w2c, K, int(W), int(H)
        except Exception as e:
            raise RuntimeError(f"Could not load COLMAP cameras from {sparse_dir}: {e}")

    raise FileNotFoundError(
        f"No transforms.json or sparse/ found in {scene_dir}. "
        "Use --scene-dir to point to the scene directory."
    )


def sh_dc_to_rgb(f_dc):
    """Convert SH DC coefficients to [0,1] RGB."""
    C0 = 0.28209479177387814
    return np.clip(f_dc * C0 + 0.5, 0, 1)


def render(gs, viewmat_t, K_t, W, H, device, render_lang=False):
    """Render either color (SH DC) or language features."""
    from gsplat import rasterization

    means   = torch.from_numpy(gs["xyz"]).to(device)
    quats   = F.normalize(torch.from_numpy(gs["quats"]).to(device), dim=-1)
    scales  = torch.exp(torch.from_numpy(gs["scales_raw"]).to(device))
    opacs   = torch.sigmoid(torch.from_numpy(gs["opacity_raw"]).to(device))

    if render_lang and gs["lang_feats"] is not None:
        colors = torch.from_numpy(gs["lang_feats"]).to(device)  # (N, 3)
    else:
        rgb = sh_dc_to_rgb(gs["f_dc"])
        colors = torch.from_numpy(rgb).to(device)

    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacs,
            colors=colors,
            viewmats=viewmat_t.unsqueeze(0),
            Ks=K_t.unsqueeze(0),
            width=W, height=H,
            sh_degree=None,
            near_plane=0.01, far_plane=1000.0,
        )
    return renders[0].cpu().numpy()  # (H, W, C)


def make_heatmap(relevance: np.ndarray, color_img: np.ndarray, alpha=0.55):
    """
    relevance: (H, W) float in [0,1]
    color_img: (H, W, 3) float in [0,1]
    Returns (H, W, 3) uint8 overlay.
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap("inferno")
    heat = cmap(relevance)[..., :3]  # (H, W, 3) float
    blended = (1 - alpha) * color_img + alpha * heat
    return (blended.clip(0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="LangSplat query")
    parser.add_argument("--lang-ply", required=True, help="Language-augmented PLY file")
    parser.add_argument("--autoencoder", required=True, help="Autoencoder checkpoint (.pth)")
    parser.add_argument("--query", required=True, help="Text query string")
    parser.add_argument("--scene-dir", required=True,
                        help="Scene directory (with transforms.json or sparse/) for camera pose")
    parser.add_argument("--output", default="langsplat_query.png", help="Output heatmap PNG")
    parser.add_argument("--relevance-threshold", type=float, default=0.0,
                        help="Minimum cosine similarity to show (0=show all)")
    args = parser.parse_args()

    ensure_gsplat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LangSplat query] device={device}  query='{args.query}'")

    print("[LangSplat query] Loading language PLY...")
    gs = load_lang_ply(args.lang_ply)
    if gs["lang_feats"] is None:
        sys.exit(1)
    N = gs["xyz"].shape[0]
    print(f"  {N:,} Gaussians")

    print("[LangSplat query] Loading autoencoder...")
    ae = load_autoencoder(args.autoencoder, device)

    print(f"[LangSplat query] Encoding query: '{args.query}'")
    text_z = encode_text_query(args.query, ae, device)  # (3,)
    text_z_np = text_z.cpu().numpy()

    print(f"[LangSplat query] Loading camera from scene: {args.scene_dir}")
    w2c, K, W, H = load_first_training_camera(args.scene_dir)
    viewmat_t = torch.from_numpy(w2c).to(device)
    K_t = torch.from_numpy(K).to(device)

    print(f"[LangSplat query] Rendering at {W}×{H}...")
    # Render colour view for background
    color_np = render(gs, viewmat_t, K_t, W, H, device, render_lang=False)
    color_np = color_np.clip(0, 1)

    # Render language feature map
    lang_np = render(gs, viewmat_t, K_t, W, H, device, render_lang=True)  # (H, W, 3)

    # Cosine similarity between each pixel's lang feature and the text query
    lang_norm = lang_np / (np.linalg.norm(lang_np, axis=-1, keepdims=True) + 1e-6)
    txt_norm = text_z_np / (np.linalg.norm(text_z_np) + 1e-6)
    relevance = (lang_norm * txt_norm[None, None, :]).sum(axis=-1)  # (H, W)

    # Threshold and normalize to [0, 1]
    relevance = np.clip(relevance, args.relevance_threshold, 1.0)
    r_min, r_max = relevance.min(), relevance.max()
    if r_max > r_min:
        relevance = (relevance - r_min) / (r_max - r_min)

    # Composite heatmap over colour
    heatmap = make_heatmap(relevance, color_np)
    Image.fromarray(heatmap).save(args.output)
    print(f"[LangSplat query] Saved heatmap: {args.output}")

    # Also save raw relevance as npy for downstream use
    npy_out = args.output.replace(".png", "_relevance.npy")
    np.save(npy_out, relevance)
    print(f"[LangSplat query] Saved relevance map: {npy_out}")


if __name__ == "__main__":
    main()
