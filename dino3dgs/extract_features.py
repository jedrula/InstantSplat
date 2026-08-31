#!/usr/bin/env python3
"""
DINOv2 feature extraction for 3DGS distillation.

Extracts DINOv2 ViT-S/14 patch features from training images, fits PCA to
compress 384D → n_components (default 8), saves compressed maps per image.

Runtime: ~0.1s/image on GPU. No SAM2, no CLIP, no segmentation.

Usage:
  python extract_features.py --image-dir DIR --output-dir DIR [--n-components 8]
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


DINO_PATCH = 14  # ViT-S/14 and ViT-B/14 both use 14px patches


def load_dino(model_name="dinov2_vits14", device="cuda"):
    model = torch.hub.load(
        "facebookresearch/dinov2", model_name,
        pretrained=True, verbose=False
    )
    model = model.to(device).eval()
    return model


def make_patch_size(w, h, max_long_side=560):
    """Return (pw, ph) divisible by DINO_PATCH, with longest side ≤ max_long_side."""
    scale = max_long_side / max(w, h)
    pw = max(DINO_PATCH, round(w * scale / DINO_PATCH) * DINO_PATCH)
    ph = max(DINO_PATCH, round(h * scale / DINO_PATCH) * DINO_PATCH)
    return pw, ph


def extract_patch_features(model, pil_img, pw, ph, device):
    """
    Resize image to (pw, ph), run DINOv2, return (ph//14, pw//14, 384) float32.
    """
    # DINOv2 ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    img = pil_img.resize((pw, ph), Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).float().to(device) / 255.0
    t = t.permute(2, 0, 1)           # (3, ph, pw)
    t = (t - mean) / std
    t = t.unsqueeze(0)                # (1, 3, ph, pw)

    with torch.no_grad():
        feats = model.forward_features(t)
        # 'x_norm_patchtokens' shape: (1, n_patches, dim)
        patch_tokens = feats["x_norm_patchtokens"]

    n_h = ph // DINO_PATCH
    n_w = pw // DINO_PATCH
    patch_tokens = patch_tokens.squeeze(0)          # (n_h*n_w, dim)
    patch_map = patch_tokens.view(n_h, n_w, -1)     # (n_h, n_w, dim)
    return patch_map.cpu().float().numpy()


def fit_pca(feature_files, n_components, device, sample_per_file=2048):
    """Fit PCA on a random subsample of all feature maps. Returns V (dim, n_components)."""
    print(f"[DINO] Fitting PCA ({n_components}D) on {len(feature_files)} feature maps...")
    rng = np.random.default_rng(42)
    chunks = []
    for fpath in feature_files:
        fm = np.load(fpath)          # (H_p, W_p, dim)
        pixels = fm.reshape(-1, fm.shape[-1])
        idx = rng.choice(len(pixels), min(sample_per_file, len(pixels)), replace=False)
        chunks.append(pixels[idx])
    X = np.concatenate(chunks, axis=0).astype(np.float32)   # (N_total, dim)
    X_t = torch.from_numpy(X).to(device)
    # Centre
    mean = X_t.mean(0)
    X_t = X_t - mean
    _, _, V = torch.pca_lowrank(X_t, q=n_components, niter=4)
    # V: (dim, n_components)
    return mean.cpu().numpy(), V.cpu().numpy()


def compress(feat_map, mean, V):
    """feat_map (N, dim) or (H, W, dim) → same leading dims, k trailing. float16."""
    shape = feat_map.shape
    pixels = feat_map.reshape(-1, shape[-1]).astype(np.float32)
    pixels -= mean[np.newaxis, :]
    compressed = pixels @ V          # (N, k)
    return compressed.reshape(*shape[:-1], -1).astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir",    required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--n-components", type=int, default=8,
                        help="PCA output dimensions (default 8)")
    parser.add_argument("--max-side",     type=int, default=560,
                        help="Longest side of images fed to DINOv2 (multiple of 14)")
    parser.add_argument("--model",        default="dinov2_vits14",
                        help="DINOv2 variant (dinov2_vits14 or dinov2_vitb14)")
    parser.add_argument("--exts",         default="jpg,jpeg,png,webp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exts = tuple(f".{e}" for e in args.exts.split(","))
    img_paths = sorted(p for p in Path(args.image_dir).iterdir()
                       if p.suffix.lower() in exts)
    if not img_paths:
        print(f"ERROR: no images in {args.image_dir}"); sys.exit(1)

    print(f"[DINO] Loading {args.model}...")
    model = load_dino(args.model, device)

    # ── Pass 1: extract raw DINOv2 features ───────────────────────────────────
    raw_dir = os.path.join(args.output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"[DINO] Extracting features from {len(img_paths)} images...")
    meta_entries = []
    pw = ph = None
    import time
    t0 = time.time()
    for i, img_path in enumerate(img_paths):
        raw_path = os.path.join(raw_dir, img_path.stem + "_dino.npy")
        if not os.path.exists(raw_path):
            img = Image.open(img_path).convert("RGB")
            if pw is None:
                pw, ph = make_patch_size(*img.size, max_long_side=args.max_side)
                print(f"  Patch grid: {pw}×{ph}px → {ph//DINO_PATCH}×{pw//DINO_PATCH} patches")
            fm = extract_patch_features(model, img, pw, ph, device)
            np.save(raw_path, fm.astype(np.float16))
        meta_entries.append({"image": str(img_path), "raw": raw_path})
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(img_paths) - i - 1) / rate
            print(f"  [{i+1}/{len(img_paths)}]  {rate:.1f} img/s  ~{remaining:.0f}s remaining")

    elapsed_p1 = time.time() - t0
    print(f"[DINO] Feature extraction: {elapsed_p1:.1f}s  ({elapsed_p1/len(img_paths):.2f}s/img)")

    # ── Pass 2: fit PCA ───────────────────────────────────────────────────────
    raw_files = [e["raw"] for e in meta_entries]
    mean, V = fit_pca(raw_files, args.n_components, device)
    pca_path = os.path.join(args.output_dir, "pca.npz")
    np.savez(pca_path, mean=mean, V=V)
    print(f"[DINO] PCA saved: {pca_path}")

    # ── Pass 3: compress all feature maps ────────────────────────────────────
    print(f"[DINO] Compressing to {args.n_components}D...")
    t2 = time.time()
    for entry in meta_entries:
        raw = np.load(entry["raw"]).astype(np.float32)
        comp = compress(raw, mean, V)                  # (H_p, W_p, k)
        comp_path = entry["raw"].replace("_dino.npy", "_dino_comp.npy").replace(raw_dir, args.output_dir)
        np.save(comp_path, comp)
        entry["compressed"] = comp_path
    print(f"[DINO] Compression: {time.time()-t2:.1f}s")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "model": args.model,
        "n_components": args.n_components,
        "patch_size": DINO_PATCH,
        "patch_grid_w": pw // DINO_PATCH if pw else None,
        "patch_grid_h": ph // DINO_PATCH if ph else None,
        "pca": pca_path,
        "images": meta_entries,
    }
    meta_path = os.path.join(args.output_dir, "dino_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[DINO] Done. {meta_path}")
    print(f"       Total: {time.time()-t0:.1f}s for {len(img_paths)} images")


if __name__ == "__main__":
    main()
