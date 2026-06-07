#!/usr/bin/env python3
"""
Extract a good initial camera from nerfstudio transforms.json.

Selects the training frame that gives a "person standing in front of the wall" view
(most central, most wall-facing). Optionally accepts --frame <name> to pin a specific
image by filename (basename match).

The camera's own col1 (physical "up" when the photo was taken) is used as the viewer
up vector — this is correct regardless of which axis nerfstudio/COLMAP chose as "up".

Applies dataparser_transforms.json (if present) so that the output is in the
PLY/training coordinate system that the Gaussian splat viewer uses.

Outputs initial_camera.json:
  position     — camera position in training space
  look_at      — point the camera faces in training space (pos + fwd * dist)
  up           — camera up direction in training space (col1 of c2w)
  source_frame — filename for debugging

Usage:
    python extract_initial_camera.py <transforms.json> [output.json] [--frame <image_name>]
"""

import json
import math
import sys
from pathlib import Path


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-8 else v


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _median(vals):
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def extract(transforms_path: Path, out_path: Path, pin_frame: str = None):
    d = json.loads(transforms_path.read_text())

    frames = d.get("frames", [])
    if not frames:
        print("No frames in transforms.json", file=sys.stderr)
        sys.exit(1)

    # Build 4×4 applied_transform (3×4 stored, last row implicit [0,0,0,1])
    at_raw = d.get("applied_transform")
    at = ([list(row) for row in at_raw] + [[0.0, 0.0, 0.0, 1.0]]) if at_raw \
        else [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

    def mat_mul_4x4(A, B):
        C = [[0.0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    # Extract per-frame position, forward, and camera-up vectors in ns_input space.
    # nerfstudio uses OpenGL convention: camera looks along -Z, so world_fwd = -col2.
    cameras = []
    for fr in frames:
        c2w = mat_mul_4x4(at, fr["transform_matrix"])
        pos    = [c2w[i][3] for i in range(3)]
        fwd    = _normalize([-c2w[i][2] for i in range(3)])   # -col2 = forward
        cam_up = _normalize([c2w[i][1] for i in range(3)])    # col1 = camera up
        cameras.append({"pos": pos, "fwd": fwd, "cam_up": cam_up,
                        "file_path": fr["file_path"]})

    # Centroid and per-axis stats
    cx = sum(c["pos"][0] for c in cameras) / len(cameras)
    cy = sum(c["pos"][1] for c in cameras) / len(cameras)
    cz = sum(c["pos"][2] for c in cameras) / len(cameras)
    centroid = [cx, cy, cz]

    xs = [c["pos"][0] for c in cameras]
    ys = [c["pos"][1] for c in cameras]
    zs = [c["pos"][2] for c in cameras]
    median_x = _median(xs)
    median_y = _median(ys)
    median_z = _median(zs)
    x_spread = max(max(xs) - min(xs), 1e-3)
    y_spread = max(max(ys) - min(ys), 1e-3)
    z_spread = max(max(zs) - min(zs), 1e-3)
    bbox_diag = math.sqrt(x_spread**2 + y_spread**2 + z_spread**2)

    # ── Frame selection ────────────────────────────────────────────────────────
    if pin_frame:
        # User-specified frame: match by basename (e.g. "frame_00040.png")
        pin_base = Path(pin_frame).name
        matched = [c for c in cameras if Path(c["file_path"]).name == pin_base]
        if not matched:
            print(f"Warning: --frame '{pin_frame}' not found; auto-selecting.", file=sys.stderr)
            pin_frame = None
        else:
            best = matched[0]
            print(f"  Using pinned frame: {best['file_path']}", flush=True)

    if not pin_frame:
        def score(cam):
            # Must face the centroid
            to_c = _normalize([centroid[i] - cam["pos"][i] for i in range(3)])
            if _dot(cam["fwd"], to_c) < 0.2:
                return float("inf")
            x_err = abs(cam["pos"][0] - median_x) / x_spread
            y_err = abs(cam["pos"][1] - median_y) / y_spread
            z_err = abs(cam["pos"][2] - median_z) / z_spread
            # Central position matters; no tilt penalty (we use cam_up directly)
            return 1.0 * x_err + 1.5 * y_err + 2.5 * z_err

        best = min(cameras, key=score)
        if score(best) == float("inf"):
            best = max(cameras,
                       key=lambda c: _dot(c["fwd"],
                                          _normalize([centroid[i] - c["pos"][i] for i in range(3)])))

    pos_ns  = best["pos"]
    fwd_ns  = best["fwd"]    # camera's actual forward direction
    up_ns   = best["cam_up"] # camera's actual physical up (col1 of c2w) — correct regardless of coord system
    look_dist = bbox_diag * 0.25

    # ── Apply dataparser_transforms.json ─────────────────────────────────────
    # Nerfstudio applies an additional transform + scale when loading training data.
    # The exported PLY is in this training space, so the viewer camera must be too.
    # Search for the most recent run's dataparser_transforms.json under pod_dir.
    pod_dir = transforms_path.parent.parent
    dt_files = sorted(pod_dir.rglob("dataparser_transforms.json"))
    if dt_files:
        dt = json.loads(dt_files[-1].read_text())   # most recent run
        T_raw = dt["transform"]                      # 3×4
        dt_scale = dt["scale"]
        R_dt = [[T_raw[i][j] for j in range(3)] for i in range(3)]
        t_dt = [T_raw[i][3] for i in range(3)]

        def apply_pt(p):
            """Transform a 3D point from ns_input → training space."""
            q = [sum(R_dt[i][j] * p[j] for j in range(3)) + t_dt[i] for i in range(3)]
            return [dt_scale * q[i] for i in range(3)]

        def apply_dir(v):
            """Rotate a direction from ns_input → training space (no scale/translation)."""
            q = [sum(R_dt[i][j] * v[j] for j in range(3)) for i in range(3)]
            return _normalize(q)

        pos      = apply_pt(pos_ns)
        fwd      = apply_dir(fwd_ns)
        up       = apply_dir(up_ns)
        look_dist = look_dist * dt_scale
        print(f"  dataparser_transforms: {dt_files[-1].relative_to(pod_dir)}", flush=True)
        print(f"  dt_scale: {dt_scale:.5f}", flush=True)
    else:
        pos  = pos_ns
        fwd  = fwd_ns
        up   = up_ns

    look_at = [pos[i] + fwd[i] * look_dist for i in range(3)]

    result = {
        "position":     [round(v, 4) for v in pos],
        "look_at":      [round(v, 4) for v in look_at],
        "up":           [round(v, 4) for v in up],
        "source_frame": best["file_path"],
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"initial_camera.json → {out_path}")
    print(f"  frame:   {best['file_path']}")
    print(f"  pos:     {result['position']}")
    print(f"  look_at: {result['look_at']}")
    print(f"  up:      {result['up']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <transforms.json> [output.json] [--frame <image_name>]")
        sys.exit(1)
    tf = Path(sys.argv[1])
    # Parse remaining args: optional positional output path and --frame flag
    rest = sys.argv[2:]
    pin = None
    out_args = []
    i = 0
    while i < len(rest):
        if rest[i] == "--frame" and i + 1 < len(rest):
            pin = rest[i + 1]
            i += 2
        else:
            out_args.append(rest[i])
            i += 1
    out = Path(out_args[0]) if out_args else tf.parent / "initial_camera.json"
    extract(tf, out, pin_frame=pin)
