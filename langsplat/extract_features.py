#!/usr/bin/env python3
"""
LangSplat Phase 1: Extract per-image CLIP language features using SAM2 segmentation.

Requires: sam2 + open_clip_torch (available in the sam2 venv)
  /workdir/sam2/venv/bin/python extract_features.py --image-dir ... --output-dir ...

For each image:
  1. Run SAM2 AutomaticMaskGenerator to get N segments
  2. For each segment: extract CLIP feature of the tight crop
  3. Build pixel-level feature map: each pixel gets the CLIP feature of its segment
     (pixels in no segment get the whole-image CLIP feature)
  4. Save as (H//4, W//4, 512) float16 .npy to save disk space

Also saves features_meta.json listing all images and their feature files.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import open_clip


SAM2_PKG = "/home/communications/workdir/sam2/venv/lib/python3.12/site-packages"
SAM2_MODELS = "/home/communications/workdir/sam2/models"


def load_sam2_generator(device):
    if SAM2_PKG not in sys.path:
        sys.path.insert(0, SAM2_PKG)
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # build_sam2 expects just the filename; hydra resolves it from pkg://sam2
    ckpt_path = os.path.join(SAM2_MODELS, "sam2_hiera_base_plus.pt")
    cfg_name = "sam2_hiera_b+.yaml"
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(SAM2_MODELS, "sam2_hiera_small.pt")
        cfg_name = "sam2_hiera_s.yaml"

    model = build_sam2(cfg_name, ckpt_path, device=device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=8,          # 8×8 = 64 prompts; fits on 8GB VRAM for 1080p
        pred_iou_thresh=0.80,
        stability_score_thresh=0.90,
        min_mask_region_area=200,
    )
    return generator


def load_clip_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_crops(clip_model, preprocess, pil_img, masks, device):
    """
    For each mask, crop the image to the mask bbox + 10% padding,
    encode with CLIP, return (N, 512) float32 features.
    Also encode the whole image as fallback.
    """
    w, h = pil_img.size
    crops = []

    # whole-image encoding first (index 0)
    crops.append(preprocess(pil_img))

    for mask_dict in masks:
        seg = mask_dict["segmentation"]  # (H, W) bool
        ys, xs = np.where(seg)
        if len(xs) == 0:
            # degenerate mask — use whole image
            crops.append(preprocess(pil_img))
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # 10% padding
        pw = max(1, int((x1 - x0) * 0.10))
        ph = max(1, int((y1 - y0) * 0.10))
        x0 = max(0, x0 - pw)
        y0 = max(0, y0 - ph)
        x1 = min(w, x1 + pw)
        y1 = min(h, y1 + ph)
        crop = pil_img.crop((x0, y0, x1, y1))
        crops.append(preprocess(crop))

    # Batch encode all crops
    batch = torch.stack(crops).to(device)
    feats = clip_model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
    return feats.cpu().float().numpy()  # (N+1, 512)


def build_feature_map(pil_img, masks, feats, out_h, out_w):
    """
    feats: (1+N_masks, 512) — index 0 is whole-image feature
    Returns (out_h, out_w, 512) float16 feature map.
    """
    feat_map = np.zeros((out_h, out_w, 512), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w), dtype=np.float32)

    orig_h, orig_w = pil_img.size[1], pil_img.size[0]
    scale_y = out_h / orig_h
    scale_x = out_w / orig_w

    for i, mask_dict in enumerate(masks):
        seg = mask_dict["segmentation"]  # (H, W) bool original res
        # Downsample mask to output resolution
        seg_small = np.array(
            Image.fromarray(seg.astype(np.uint8) * 255).resize(
                (out_w, out_h), Image.NEAREST
            )
        ) > 128
        feat = feats[i + 1]  # feature for this segment (offset by 1 for whole-image)
        feat_map[seg_small] += feat
        weight_map[seg_small] += 1.0

    # Pixels covered by at least one mask: normalize by count
    covered = weight_map > 0
    feat_map[covered] /= weight_map[covered, np.newaxis]

    # Uncovered pixels: use whole-image feature
    feat_map[~covered] = feats[0]

    # L2 normalize each pixel's feature
    norms = np.linalg.norm(feat_map, axis=-1, keepdims=True).clip(min=1e-6)
    feat_map = feat_map / norms

    return feat_map.astype(np.float16)


def process_image(img_path, sam_gen, clip_model, preprocess, device, out_dir, scale=4):
    img_name = Path(img_path).stem
    out_path = os.path.join(out_dir, f"{img_name}_lang.npy")
    if os.path.exists(out_path):
        return out_path  # skip already processed

    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size
    out_w, out_h = max(1, w // scale), max(1, h // scale)

    # Run SAM2 segmentation
    # Cap resolution for SAM2 to avoid OOM on large images (≥1080p)
    SAM_MAX = 1024
    sam_w, sam_h = pil_img.size
    if max(sam_w, sam_h) > SAM_MAX:
        scale = SAM_MAX / max(sam_w, sam_h)
        sam_pil = pil_img.resize((int(sam_w * scale), int(sam_h * scale)), Image.BILINEAR)
    else:
        sam_pil = pil_img
    img_np = np.array(sam_pil)
    masks = sam_gen.generate(img_np)
    # Scale masks back to original image coordinates if we downscaled
    if sam_pil is not pil_img:
        sx = w / sam_pil.size[0]
        sy = h / sam_pil.size[1]
        scaled_masks = []
        for m in masks:
            seg_small = m["segmentation"]
            seg_full = np.array(
                Image.fromarray(seg_small.astype(np.uint8) * 255).resize(
                    (w, h), Image.NEAREST
                )
            ) > 128
            m = dict(m)
            m["segmentation"] = seg_full
            scaled_masks.append(m)
        masks = scaled_masks

    # Encode all crops with CLIP
    feats = encode_crops(clip_model, preprocess, pil_img, masks, device)

    # Build feature map
    feat_map = build_feature_map(pil_img, masks, feats, out_h, out_w)
    np.save(out_path, feat_map)
    torch.cuda.empty_cache()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="LangSplat feature extraction")
    parser.add_argument("--image-dir", required=True, help="Directory of training images")
    parser.add_argument("--output-dir", required=True, help="Directory to write feature .npy files")
    parser.add_argument("--scale", type=int, default=4, help="Spatial downscale factor (default 4)")
    parser.add_argument("--exts", default="jpg,jpeg,png,webp", help="Image extensions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LangSplat] device={device}  scale=1/{args.scale}")

    print("[LangSplat] Loading SAM2...")
    sam_gen = load_sam2_generator(device)

    print("[LangSplat] Loading CLIP (ViT-B-32 / OpenAI)...")
    clip_model, preprocess, _ = load_clip_model(device)

    exts = tuple(f".{e}" for e in args.exts.split(","))
    img_paths = sorted(
        p for p in Path(args.image_dir).iterdir()
        if p.suffix.lower() in exts
    )
    if not img_paths:
        print(f"ERROR: no images found in {args.image_dir}")
        sys.exit(1)

    print(f"[LangSplat] Processing {len(img_paths)} images...")
    meta = {"images": []}
    for i, img_path in enumerate(img_paths):
        out_path = process_image(
            str(img_path), sam_gen, clip_model, preprocess, device,
            args.output_dir, scale=args.scale
        )
        meta["images"].append({"image": str(img_path), "features": out_path})
        print(f"  [{i+1}/{len(img_paths)}] {img_path.name} → {os.path.basename(out_path)}")

    meta["scale"] = args.scale
    meta["clip_model"] = "ViT-B-32"
    meta["clip_dim"] = 512
    meta_path = os.path.join(args.output_dir, "features_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[LangSplat] Done. Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
