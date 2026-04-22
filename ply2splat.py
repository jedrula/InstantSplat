"""
Convert a 3DGS .ply file to the .splat binary format readable by WebGL viewers
(antimatter15/splat, gsplat.js, SuperSplat, etc.)

.splat format per Gaussian (32 bytes):
  x, y, z         : float32  (12 bytes)  position
  scale_0..2      : uint8    (3 bytes)   log-scale, mapped to [0,1]
  r, g, b, a      : uint8    (4 bytes)   color + opacity (sigmoid of raw opacity)
  rot_0..3        : uint8    (4 bytes)   quaternion, normalised, mapped to [-1,1]
  padding          : uint8   (5 bytes — wait, let's count: 12+3+4+4 = 23... )

Actually the canonical format is 32 bytes:
  x,y,z           float32  12 bytes
  scale_0,1,2     float32  12 bytes  (log-scale, stored as-is but we keep float)
  ...

No — the correct antimatter15 .splat format is exactly 32 bytes per splat:
  position:  3 × float32  = 12 bytes
  scale:     3 × float32  = 12 bytes  (actually stored as exp(scale))
  color:     4 × uint8    =  4 bytes  (rgba, opacity = sigmoid(raw_opacity))
  rotation:  4 × uint8    =  4 bytes  (normalised quaternion mapped to 0-255)

Total: 12+12+4+4 = 32 bytes ✓
"""

import struct
import numpy as np
import sys
from pathlib import Path


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_ply_gaussian(ply_path):
    with open(ply_path, "rb") as f:
        # parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        props = []
        n_verts = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property float "):
                props.append(line.split()[-1])

        print(f"  {n_verts} Gaussians, {len(props)} properties")
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape(n_verts, len(props))

    prop_idx = {name: i for i, name in enumerate(props)}
    return data, prop_idx, n_verts


def ply_to_splat(ply_path: str, out_path: str):
    print(f"Reading {ply_path} ...")
    data, idx, N = read_ply_gaussian(ply_path)

    x = data[:, idx["x"]]
    y = data[:, idx["y"]]
    z = data[:, idx["z"]]

    # Scales (log-scale → actual scale)
    sx = np.exp(data[:, idx["scale_0"]])
    sy = np.exp(data[:, idx["scale_1"]])
    sz = np.exp(data[:, idx["scale_2"]])

    # Color from DC spherical harmonic coefficient
    SH_C0 = 0.28209479177387814
    r = (0.5 + SH_C0 * data[:, idx["f_dc_0"]]).clip(0, 1)
    g = (0.5 + SH_C0 * data[:, idx["f_dc_1"]]).clip(0, 1)
    b = (0.5 + SH_C0 * data[:, idx["f_dc_2"]]).clip(0, 1)

    # Opacity
    opacity = sigmoid(data[:, idx["opacity"]])

    # Rotation quaternion (w, x, y, z stored in ply as rot_0..3)
    rw = data[:, idx["rot_0"]].copy()
    rx = data[:, idx["rot_1"]].copy()
    ry = data[:, idx["rot_2"]].copy()
    rz = data[:, idx["rot_3"]].copy()
    norm = np.sqrt(rw**2 + rx**2 + ry**2 + rz**2) + 1e-8
    rw /= norm; rx /= norm; ry /= norm; rz /= norm

    # Sort by opacity descending (improves rendering performance)
    order = np.argsort(-opacity)

    print(f"Writing {out_path} ...")
    with open(out_path, "wb") as f:
        for i in order:
            # position
            f.write(struct.pack("<fff", x[i], y[i], z[i]))
            # scale (stored as actual scale, not log)
            f.write(struct.pack("<fff", sx[i], sy[i], sz[i]))
            # rgba uint8
            ri = int(r[i] * 255)
            gi = int(g[i] * 255)
            bi = int(b[i] * 255)
            ai = int(opacity[i] * 255)
            f.write(struct.pack("<BBBB", ri, gi, bi, ai))
            # quaternion uint8 (mapped from [-1,1] to [0,255])
            def q2u(v): return int((v * 128 + 128)).clip(0, 255)
            f.write(struct.pack("<BBBB",
                int(np.clip(rw[i] * 128 + 128, 0, 255)),
                int(np.clip(rx[i] * 128 + 128, 0, 255)),
                int(np.clip(ry[i] * 128 + 128, 0, 255)),
                int(np.clip(rz[i] * 128 + 128, 0, 255)),
            ))

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Done. {N} splats → {size_mb:.1f} MB at {out_path}")


if __name__ == "__main__":
    ply = sys.argv[1] if len(sys.argv) > 1 else \
        "output_infer/climbing_section1/point_cloud/iteration_1000/point_cloud.ply"
    splat = sys.argv[2] if len(sys.argv) > 2 else ply.replace(".ply", ".splat")
    ply_to_splat(ply, splat)
