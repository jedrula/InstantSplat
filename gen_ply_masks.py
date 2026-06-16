#!/usr/bin/env python3
"""
Generate per-frame binary masks from a nerfstudio transforms.json + PLY point cloud.
Pixels where PLY points project are white (255); everything else is black (0).
Masks are dilated to fill gaps between projected points.

Usage:
    python gen_ply_masks.py --transforms TRANSFORMS_JSON --ply PLY_PATH --out MASK_DIR
                            [--dilation RADIUS]  [--min-depth MIN_DEPTH]
"""
import argparse, json, os
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.ndimage import binary_dilation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--transforms", required=True)
    p.add_argument("--ply", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dilation", type=int, default=15,
                   help="Dilation radius in pixels to fill gaps (default 15)")
    p.add_argument("--min-depth", type=float, default=0.05,
                   help="Minimum positive depth to accept a projection (default 0.05)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.transforms) as f:
        meta = json.load(f)

    print(f"Loading PLY: {args.ply}")
    ply = PlyData.read(args.ply)
    pts = np.stack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]], axis=1)
    print(f"  {len(pts):,} points")

    fl_x = meta["fl_x"]
    fl_y = meta["fl_y"]
    cx = meta["cx"]
    cy = meta["cy"]
    W = int(meta["w"])
    H = int(meta["h"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    struct = np.ones((2 * args.dilation + 1, 2 * args.dilation + 1), dtype=bool)

    frames = meta["frames"]
    for idx, fr in enumerate(frames):
        c2w = np.array(fr["transform_matrix"])
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w

        pts_cam = (R_w2c @ pts.T).T + t_w2c
        z = pts_cam[:, 2]
        valid = z > args.min_depth

        x_px = (pts_cam[valid, 0] / z[valid] * fl_x + cx).astype(np.int32)
        y_px = (pts_cam[valid, 1] / z[valid] * fl_y + cy).astype(np.int32)

        in_bounds = (x_px >= 0) & (x_px < W) & (y_px >= 0) & (y_px < H)
        x_px = x_px[in_bounds]
        y_px = y_px[in_bounds]

        mask = np.zeros((H, W), dtype=bool)
        if len(x_px) > 0:
            mask[y_px, x_px] = True
            if args.dilation > 0:
                mask = binary_dilation(mask, structure=struct)

        img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        # Keep same extension as source image so gsplat ColmapParser finds the mask
        # by exact filename match. JPEG at quality=95 is fine; .astype(bool) thresholds.
        name = Path(fr["file_path"]).name
        ext = Path(name).suffix.lower()
        save_kwargs = {"quality": 95} if ext in (".jpg", ".jpeg") else {}
        img.save(out_dir / name, **save_kwargs)

        pct = mask.sum() / (W * H) * 100
        if idx % 10 == 0 or pct < 5:
            print(f"  [{idx:03d}/{len(frames)}] {name}: {pct:.1f}% masked")

    print(f"Done. Masks written to {out_dir}/")


if __name__ == "__main__":
    main()
