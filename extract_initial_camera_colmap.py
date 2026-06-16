#!/usr/bin/env python3
"""
extract_initial_camera_colmap.py — Generate initial_camera.json from COLMAP sparse model.

Selects the most central, wall-facing camera using the same scoring as
extract_initial_camera.py (nerfstudio variant). Works with any COLMAP sparse/0/
directory containing cameras.txt + images.txt.

Output format is identical to extract_initial_camera.py so both Vue and React
SplatViewer components can consume it unchanged.

COLMAP uses OpenCV convention (camera +Z into scene, +Y down in image).
  - forward in world = R_w2c.T @ [0, 0, 1]  = c2w[:, 2]
  - "up" in world    = R_w2c.T @ [0, -1, 0] = -c2w[:, 1]   (Y is down → negate)

Usage:
    python extract_initial_camera_colmap.py <sparse/0/> [output.json] [--frame <name>]
"""

import json
import math
import sys
from pathlib import Path


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-8 else list(v)


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def quat_to_rot(qw, qx, qy, qz):
    """Unit quaternion → 3×3 rotation matrix (world-to-camera)."""
    q = [qw, qx, qy, qz]
    n = math.sqrt(sum(x*x for x in q))
    qw, qx, qy, qz = [x/n for x in q]
    return [
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ]


def mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(3)) for i in range(3)]


def extract(sparse_dir: Path, out_path: Path, pin_frame: str = None):
    # ── read cameras.txt ──────────────────────────────────────────────────────
    cam_params = {}
    with open(sparse_dir / "cameras.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cam_params[int(p[0])] = {"model": p[1], "w": int(p[2]), "h": int(p[3]),
                                      "params": list(map(float, p[4:]))}

    # ── read images.txt ───────────────────────────────────────────────────────
    cameras = []
    with open(sparse_dir / "images.txt") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        qw, qx, qy, qz = map(float, p[1:5])
        tx, ty, tz      = map(float, p[5:8])
        name            = p[9]

        R_w2c = quat_to_rot(qw, qx, qy, qz)   # world → camera
        t_w2c = [tx, ty, tz]

        # camera world position: -R_w2c.T @ t_w2c
        pos = [-sum(R_w2c[j][k] * t_w2c[j] for j in range(3)) for k in range(3)]

        # forward in world: c2w[:, 2] = R_w2c.T @ [0,0,1]  (COLMAP +Z into scene)
        fwd = _normalize([R_w2c[j][2] for j in range(3)])

        # physical "up" in world: -c2w[:, 1] = -R_w2c.T @ [0,1,0]
        # (image Y is down in OpenCV → negate to get visual "up")
        up = _normalize([-R_w2c[j][1] for j in range(3)])

        cameras.append({"pos": pos, "fwd": fwd, "up": up, "name": name})
        i += 2

    if not cameras:
        print("No images found in images.txt", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(cameras)} cameras loaded from {sparse_dir}")

    # ── centroid + spread ─────────────────────────────────────────────────────
    cx = sum(c["pos"][0] for c in cameras) / len(cameras)
    cy = sum(c["pos"][1] for c in cameras) / len(cameras)
    cz = sum(c["pos"][2] for c in cameras) / len(cameras)
    centroid = [cx, cy, cz]

    xs = [c["pos"][0] for c in cameras]
    ys = [c["pos"][1] for c in cameras]
    zs = [c["pos"][2] for c in cameras]
    median_x, median_y, median_z = _median(xs), _median(ys), _median(zs)
    x_spread = max(max(xs) - min(xs), 1e-3)
    y_spread = max(max(ys) - min(ys), 1e-3)
    z_spread = max(max(zs) - min(zs), 1e-3)
    bbox_diag = math.sqrt(x_spread**2 + y_spread**2 + z_spread**2)

    # ── frame selection ───────────────────────────────────────────────────────
    best = None
    if pin_frame:
        pin_base = Path(pin_frame).name
        matched  = [c for c in cameras if Path(c["name"]).name == pin_base]
        if matched:
            best = matched[0]
            print(f"  Using pinned frame: {best['name']}")
        else:
            print(f"  Warning: --frame '{pin_frame}' not found; auto-selecting.", file=sys.stderr)

    if best is None:
        def score(cam):
            to_c = _normalize([centroid[i] - cam["pos"][i] for i in range(3)])
            if _dot(cam["fwd"], to_c) < 0.2:   # must broadly face centroid
                return float("inf")
            x_err = abs(cam["pos"][0] - median_x) / x_spread
            y_err = abs(cam["pos"][1] - median_y) / y_spread
            z_err = abs(cam["pos"][2] - median_z) / z_spread
            return 1.0 * x_err + 1.5 * y_err + 2.5 * z_err

        best = min(cameras, key=score)
        if score(best) == float("inf"):
            # fallback: camera most aligned with centroid
            best = max(cameras,
                       key=lambda c: _dot(c["fwd"],
                                          _normalize([centroid[i] - c["pos"][i] for i in range(3)])))

    look_dist = bbox_diag * 0.25
    look_at   = [best["pos"][i] + best["fwd"][i] * look_dist for i in range(3)]

    result = {
        "position":     [round(v, 4) for v in best["pos"]],
        "look_at":      [round(v, 4) for v in look_at],
        "up":           [round(v, 4) for v in best["up"]],
        "source_frame": best["name"],
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    print(f"  frame:   {result['source_frame']}")
    print(f"  pos:     {result['position']}")
    print(f"  look_at: {result['look_at']}")
    print(f"  up:      {result['up']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sparse/0/> [output.json] [--frame <name>]")
        sys.exit(1)

    sparse = Path(sys.argv[1])
    rest   = sys.argv[2:]
    pin    = None
    out_args = []
    i = 0
    while i < len(rest):
        if rest[i] == "--frame" and i + 1 < len(rest):
            pin = rest[i + 1]; i += 2
        else:
            out_args.append(rest[i]); i += 1

    out = Path(out_args[0]) if out_args else sparse.parent.parent / "initial_camera.json"
    extract(sparse, out, pin_frame=pin)
