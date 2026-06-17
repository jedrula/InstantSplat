#!/usr/bin/env python3
"""
expand_glomap_images.py — Expand GLOMAP's sparse images.txt to full COLMAP format.

GLOMAP only records matched/triangulated observations in images.txt, not all
detected keypoints. COLMAP tools (image_registrator etc.) expect ALL keypoints
listed, with point3D_id=-1 for unmatched ones.

This script reads the database (all keypoints with exact positions) and the
GLOMAP images.txt (partial observations with 3D point assignments), merges
them per-image, and writes an expanded images.txt where:
- Every database keypoint appears (in database order, preserving indices)
- Matched keypoints carry their point3D_id from GLOMAP
- Unmatched keypoints have point3D_id=-1

Usage:
    python expand_glomap_images.py <database.db> <sparse/0/> [--in-place]
"""
import sqlite3
import sys
import shutil
import struct
from pathlib import Path
from scipy.spatial import KDTree
import numpy as np


def read_keypoints_from_db(db_path: Path):
    """Returns {image_id: np.array(N,2) of (x,y) positions}"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT image_id, rows, cols, data FROM keypoints").fetchall()
    conn.close()
    result = {}
    for img_id, nrows, ncols, data in rows:
        if nrows == 0:
            result[img_id] = np.zeros((0, 2), dtype=np.float32)
            continue
        arr = np.frombuffer(data, dtype=np.float32).reshape(nrows, ncols)
        result[img_id] = arr[:, :2]  # x, y (first two columns)
    return result


def parse_images_txt(images_txt: Path):
    """Returns {img_id: {'header_line': str, 'obs': list of (x,y,pt3d_id)}}"""
    result = {}
    with open(images_txt) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        header = lines[i]
        img_id = int(header.split()[0])
        obs_vals = list(map(float, lines[i+1].split()))
        obs = [(obs_vals[j], obs_vals[j+1], int(obs_vals[j+2]))
               for j in range(0, len(obs_vals), 3)]
        result[img_id] = {"header": header.rstrip(), "obs": obs}
        i += 2
    return result


def main():
    in_place = "--in-place" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <database.db> <sparse/0/> [--in-place]")

    db_path    = Path(args[0])
    sparse_dir = Path(args[1])
    images_txt = sparse_dir / "images.txt"

    if not images_txt.exists():
        sys.exit(f"images.txt not found in {sparse_dir}")

    print("Loading database keypoints...")
    db_kps = read_keypoints_from_db(db_path)
    total_kps = sum(len(v) for v in db_kps.values())
    print(f"  {len(db_kps)} images, {total_kps} total keypoints")

    print("Parsing GLOMAP images.txt...")
    glomap_imgs = parse_images_txt(images_txt)
    total_obs = sum(len(v["obs"]) for v in glomap_imgs.values())
    print(f"  {len(glomap_imgs)} images, {total_obs} observations")

    # ── Build output ──────────────────────────────────────────────────────────
    if not in_place:
        backup = images_txt.with_suffix(".txt.bak")
        shutil.copy2(images_txt, backup)
        print(f"  Backed up to {backup}")

    out_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(glomap_imgs)}, mean observations per image: "
        f"{total_kps / max(len(glomap_imgs), 1):.1f}",
    ]

    expanded = 0
    for img_id, data in sorted(glomap_imgs.items()):
        out_lines.append(data["header"])

        db_xy = db_kps.get(img_id)
        if db_xy is None or len(db_xy) == 0:
            # No keypoints in DB for this image — keep GLOMAP obs as-is
            obs_line = " ".join(f"{x} {y} {p}" for x, y, p in data["obs"])
            out_lines.append(obs_line)
            continue

        # Build KDTree on GLOMAP observations to find point3D_id for each DB kp
        glomap_obs = data["obs"]
        if glomap_obs:
            obs_xy = np.array([(x, y) for x, y, _ in glomap_obs], dtype=np.float64)
            obs_pt3d = np.array([p for _, _, p in glomap_obs], dtype=np.int64)
            tree = KDTree(obs_xy)
        else:
            tree = None

        # For each DB keypoint, find matching GLOMAP observation (exact position)
        parts = []
        for kp_x, kp_y in db_xy:
            pt3d_id = -1
            if tree is not None:
                dist, idx = tree.query([kp_x, kp_y])
                if dist < 1.5:  # within 1.5px → same keypoint
                    pt3d_id = int(obs_pt3d[idx])
            parts.append(f"{kp_x:.4f} {kp_y:.4f} {pt3d_id}")

        expanded += 1
        out_lines.append(" ".join(parts))

    images_txt.write_text("\n".join(out_lines) + "\n")
    print(f"Expanded {expanded}/{len(glomap_imgs)} images → {images_txt}")


if __name__ == "__main__":
    main()
