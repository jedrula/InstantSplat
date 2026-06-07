#!/usr/bin/env python3
"""
Synthesise cameras.txt for RealityScan COLMAP export.

RS COLMAP writer (format {280B11A4}) writes images.txt + points3D.txt but
omits cameras.txt.  We derive intrinsics from the RS CSV export
(Internal/External Camera Parameters, format {0CA18733}) which has f_35mm,
px_norm, py_norm per registered image, then write one PINHOLE entry per
unique CAMERA_ID in images.txt.

Usage:
    python rs_make_cameras_txt.py images.txt intrinsics.csv images_dir cameras.txt
"""

import csv
import statistics
import struct
import sys
from pathlib import Path


def parse_camera_ids(images_txt_path):
    """Return sorted unique CAMERA_IDs from images.txt."""
    ids = set()
    skip_next = False
    with open(images_txt_path) as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith('#'):
                continue
            if skip_next:
                skip_next = False
                continue
            parts = line.split()
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            ids.add(int(parts[8]))
            skip_next = True
    return sorted(ids)


def parse_csv_intrinsics(csv_path):
    """Return (median_f35, median_px, median_py) from RS CSV export.

    CSV header (may start with #):
        name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2
    """
    f35_vals, px_vals, py_vals = [], [], []
    with open(csv_path, newline='') as f:
        header_line = f.readline().lstrip('#').strip()
        headers = [h.strip() for h in header_line.split(',')]
        reader = csv.DictReader(f, fieldnames=headers)
        for row in reader:
            f35_vals.append(float(row['f_35mm']))
            px_vals.append(float(row['px_norm']))
            py_vals.append(float(row['py_norm']))
    return statistics.median(f35_vals), statistics.median(px_vals), statistics.median(py_vals)


def image_size_png(path):
    """Return (width, height) from PNG IHDR without PIL."""
    with open(path, 'rb') as f:
        f.read(16)  # 8-byte sig + 4-byte len + 4-byte 'IHDR'
        w, h = struct.unpack('>II', f.read(8))
    return w, h


def main(images_txt, csv_path, images_dir, cameras_txt_out):
    camera_ids = parse_camera_ids(images_txt)
    print(f"  camera IDs: {camera_ids[:5]}{'...' if len(camera_ids) > 5 else ''} ({len(camera_ids)} total)")

    f35, px_norm, py_norm = parse_csv_intrinsics(csv_path)
    print(f"  CSV: f_35mm={f35:.3f}  px_norm={px_norm:.5f}  py_norm={py_norm:.5f}")

    sample = next(Path(images_dir).glob("*.png"))
    width, height = image_size_png(sample)
    print(f"  image size: {width}x{height}")

    # RealityCapture: f_35mm = f_normalised * 36, f_normalised = f_px / max(W,H)
    max_dim = max(width, height)
    f_px = f35 / 36.0 * max_dim
    cx = width  / 2.0 + px_norm * max_dim
    cy = height / 2.0 + py_norm * max_dim
    print(f"  → f_px={f_px:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    with open(cameras_txt_out, 'w') as out:
        out.write("# Camera list with one line of data per camera:\n")
        out.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        out.write(f"# Number of cameras: {len(camera_ids)}\n")
        for cam_id in camera_ids:
            out.write(f"{cam_id} PINHOLE {width} {height} {f_px:.6f} {f_px:.6f} {cx:.6f} {cy:.6f}\n")

    print(f"  wrote cameras.txt with {len(camera_ids)} entries → {cameras_txt_out}")


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: rs_make_cameras_txt.py images.txt intrinsics.csv images_dir cameras.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
