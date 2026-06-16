#!/usr/bin/env python3
"""
Convert a nerfstudio transforms.json to COLMAP text sparse/0/ format,
then write binary for fast loading by gsplat / brush / pgsr.

Usage:
    python ns_to_colmap.py <transforms_json> <output_sparse_dir> [--ply <ply_path>]

Output: <output_sparse_dir>/cameras.{txt,bin}, images.{txt,bin}, points3D.{txt,bin}

Coordinate convention:
  RealityScan (and some other exporters) write transforms.json in OpenCV/COLMAP convention
  (X right, Y down, Z forward) rather than the OpenGL convention (X right, Y up, Z backward).
  COLMAP w2c = inv(c2w): R_w2c = R_c2w.T, t_w2c = -R_w2c @ t_c2w.
  No axis-flip is needed for RealityScan exports.
"""
import argparse, json, struct
from pathlib import Path

import numpy as np


def _rot_to_quat_wxyz(R: np.ndarray) -> tuple:
    """Rotation matrix → (qw, qx, qy, qz) following COLMAP convention."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (w, x, y, z)


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

    # ── cameras.txt ──────────────────────────────────────────────────────────
    cameras_txt = out / "cameras.txt"
    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fl_x} {fl_y} {cx} {cy}\n")

    # ── images.txt ───────────────────────────────────────────────────────────
    # RealityScan exports use OpenCV/COLMAP convention (Z-forward, no Y-flip).
    # COLMAP w2c = inv(c2w): R_w2c = R_c2w.T, t_w2c = -R_w2c @ t_c2w
    images_txt = out / "images.txt"
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, frame in enumerate(d["frames"]):
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            R_c2w = c2w[:3, :3]
            t_c2w = c2w[:3, 3]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            qw, qx, qy, qz = _rot_to_quat_wxyz(R_w2c)
            fname = Path(frame["file_path"]).name
            image_id = idx + 1
            f.write(f"{image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                    f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {fname}\n")
            f.write("\n")  # empty POINTS2D line

    # ── points3D.txt ─────────────────────────────────────────────────────────
    ply_file = ply_path
    if ply_file is None:
        rel = d.get("ply_file_path")
        if rel:
            ply_file = str(ns_dir / rel)

    points3d_txt = out / "points3D.txt"
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
        with open(points3d_txt, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
            for pid, (p, c) in enumerate(zip(xyz, rgb)):
                f.write(f"{pid+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{c[0]} {c[1]} {c[2]} 0.0\n")
    else:
        print("  No PLY found — writing empty points3D")
        with open(points3d_txt, "w") as f:
            f.write("# 3D point list\n# Number of points: 0\n")

    # ── text → binary ─────────────────────────────────────────────────────────
    print("  Converting text → binary...")
    import pycolmap
    rec = pycolmap.Reconstruction()
    rec.read_text(str(out))
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
