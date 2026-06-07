#!/usr/bin/env python3
"""
Convert a PLY sparse point cloud (RealityScan exportSparsePointCloud output)
to COLMAP binary points3D.bin with track_length=0.

Points with empty tracks are valid COLMAP format; pycolmap loads them fine and
3DGS trainers use XYZ positions only for Gaussian initialisation.

Usage: python rs_ply_to_points3d.py input.ply output/points3D.bin
"""
import sys
import struct
from pathlib import Path


def convert(ply_path: str, out_path: str) -> int:
    try:
        import numpy as np
        from plyfile import PlyData
        ply = PlyData.read(ply_path)
        verts = ply['vertex']
        names = verts.data.dtype.names
        xs = verts['x'].astype(float)
        ys = verts['y'].astype(float)
        zs = verts['z'].astype(float)
        rs = verts['red'].astype('uint8')   if 'red'   in names else [128] * len(xs)
        gs = verts['green'].astype('uint8') if 'green' in names else [128] * len(xs)
        bs = verts['blue'].astype('uint8')  if 'blue'  in names else [128] * len(xs)
        pts = list(zip(xs, ys, zs, rs, gs, bs))
    except ImportError:
        pts = _parse_ply_ascii(ply_path)

    n = len(pts)
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<Q', n))
        for i, (x, y, z, r, g, b) in enumerate(pts):
            f.write(struct.pack('<QdddBBBdQ',
                i + 1,              # point3D_id (1-based)
                float(x), float(y), float(z),
                int(r), int(g), int(b),
                0.0,                # reprojection error
                0,                  # track_length = 0
            ))
    return n


def _parse_ply_ascii(ply_path: str):
    pts = []
    in_header = True
    with open(ply_path, 'r', errors='replace') as f:
        for line in f:
            if in_header:
                if line.strip() == 'end_header':
                    in_header = False
                continue
            p = line.split()
            if len(p) >= 3:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                r = int(p[3]) if len(p) > 3 else 128
                g = int(p[4]) if len(p) > 4 else 128
                b = int(p[5]) if len(p) > 5 else 128
                pts.append((x, y, z, r, g, b))
    return pts


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.ply output/points3D.bin", file=sys.stderr)
        sys.exit(1)
    ply, out = sys.argv[1], sys.argv[2]
    if not Path(ply).exists():
        print(f"Error: PLY not found: {ply}", file=sys.stderr)
        sys.exit(1)
    n = convert(ply, out)
    print(f"    → Wrote {n} points to {out}")
