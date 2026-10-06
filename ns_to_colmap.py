#!/usr/bin/env python3
"""
Convert a nerfstudio transforms.json to a binary COLMAP sparse/0/ model
for gsplat / brush.

Usage:
    python ns_to_colmap.py <transforms_json> <output_sparse_dir> [--ply <ply_path>]

Output: <output_sparse_dir>/{cameras,images,points3D,rigs,frames}.bin

Coordinate convention:
  RealityScan (and some other exporters) write transforms.json in OpenCV/COLMAP convention
  (X right, Y down, Z forward) rather than the OpenGL convention (X right, Y up, Z backward).
  COLMAP w2c = inv(c2w): R_w2c = R_c2w.T, t_w2c = -R_w2c @ t_c2w.
  No axis-flip is needed for RealityScan exports.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pycolmap


def _read_ply_xyz_rgb(path: str):
    with open(path, "rb") as f:
        props = []
        n_verts = 0
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        dtype_map = {
            "float": "<f4", "float32": "<f4", "double": "<f8",
            "uchar": "u1", "uint8": "u1",
            "short": "<i2", "int": "<i4", "uint": "<u4",
        }
        dt = np.dtype([(name, dtype_map[typ]) for typ, name in props])
        data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    try:
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    except ValueError:
        rgb = np.zeros((len(data), 3), dtype=np.uint8)
    return xyz, rgb


def ns_to_colmap(transforms_json: str, output_dir: str, ply_path: str | None = None,
                 max_points: int = 100_000):
    with open(transforms_json) as f:
        d = json.load(f)

    ns_dir = Path(transforms_json).parent
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    w   = int(d["w"])
    h   = int(d["h"])
    fl_x = float(d["fl_x"])
    fl_y = float(d.get("fl_y", fl_x))
    cx  = float(d["cx"])
    cy  = float(d["cy"])

    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera(
        model="PINHOLE", width=w, height=h, params=[fl_x, fl_y, cx, cy], camera_id=1))

    # RealityScan exports use OpenCV/COLMAP convention (Z-forward, no Y-flip).
    # COLMAP w2c = inv(c2w): R_w2c = R_c2w.T, t_w2c = -R_w2c @ t_c2w
    for idx, frame in enumerate(d["frames"]):
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        R_w2c = c2w[:3, :3].T
        t_w2c = -R_w2c @ c2w[:3, 3]
        image = pycolmap.Image(name=Path(frame["file_path"]).name, camera_id=1, image_id=idx + 1)
        rec.add_image_with_trivial_frame(image, pycolmap.Rigid3d(np.hstack([R_w2c, t_w2c[:, None]])))

    ply_file = ply_path
    if ply_file is None:
        rel = d.get("ply_file_path")
        if rel:
            ply_file = str(ns_dir / rel)

    if ply_file and Path(ply_file).exists():
        print(f"  Loading point cloud from {ply_file}...")
        xyz, rgb = _read_ply_xyz_rgb(ply_file)
        if len(xyz) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(xyz), max_points, replace=False)
            xyz, rgb = xyz[idx], rgb[idx]
            print(f"  Subsampled to {len(xyz):,} / {len(xyz):,} points (max_points={max_points})")
        else:
            print(f"  {len(xyz):,} points")
        for p, c in zip(xyz, rgb):
            rec.add_point3D(p, pycolmap.Track(), c)
    else:
        print("  No PLY found — writing empty points3D")

    rec.write_binary(str(out))
    print(f"  Done: {rec.num_cameras()} cam, {rec.num_images()} images, "
          f"{rec.num_points3D():,} points → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("transforms_json")
    ap.add_argument("output_dir")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--max-points", type=int, default=100_000,
                    help="Max PLY points to keep (0 = no limit, default 100000)")
    args = ap.parse_args()
    ns_to_colmap(args.transforms_json, args.output_dir, args.ply,
                 max_points=args.max_points if args.max_points > 0 else 10**9)
