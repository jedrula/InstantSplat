#!/usr/bin/env python3
"""
Neural feature matching → COLMAP database → GLOMAP global SfM (or COLMAP incremental).

Sparse matchers (keypoint + descriptor + LightGlue):
  aliked+lightglue      — ALIKED-n16 + LightGlue  (default, best so far)
  disk+lightglue        — DISK + LightGlue
  superpoint+lightglue  — SuperPoint + LightGlue

Semi-dense (LoFTR, detect+match simultaneously):
  loftr                 — LoFTR outdoor weights
  loftr_indoor          — LoFTR indoor weights (for climbing walls)

Usage:
    python glomap_hloc.py <image_dir> <output_dir>
                          [--matcher aliked+lightglue|disk+lightglue|superpoint+lightglue|loftr|loftr_indoor]
                          [--mapper glomap|colmap]
                          [--focal-length F]        prior focal length in pixels
                          [--colmap-bin /path/to/colmap]
                          [--sequential-overlap N]

Outputs:
    <output_dir>/sparse/0/  — cameras.bin + images.bin + points3D.bin
    <output_dir>/database.db — COLMAP database with features + matches
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pycolmap
from hloc import extract_features, match_dense, match_features, pairs_from_exhaustive
from hloc.reconstruction import (
    create_empty_db,
    estimation_and_geometric_verification,
    get_image_ids,
    import_features,
    import_images,
    import_matches,
)

# ── Sparse matcher configs ────────────────────────────────────────────────────
SPARSE_FEATURE_CONFS = {
    "aliked+lightglue":     "aliked-n16",
    "disk+lightglue":       "disk",
    "superpoint+lightglue": "superpoint_aachen",
}

# ── Semi-dense matcher configs ────────────────────────────────────────────────
DENSE_CONFS = {
    "loftr": match_dense.confs["loftr"],
    "loftr_indoor": {
        **match_dense.confs["loftr"],
        "output": "matches-loftr-indoor",
        "model": {"name": "loftr", "weights": "indoor"},
    },
}

ALL_MATCHERS = list(SPARSE_FEATURE_CONFS) + list(DENSE_CONFS)


def make_sequential_pairs(image_list: list[str], overlap: int, out_path: Path):
    """Write sequential pairs (each frame paired with next `overlap` frames)."""
    with open(out_path, "w") as f:
        n = len(image_list)
        for i in range(n):
            for j in range(i + 1, min(i + overlap + 1, n)):
                f.write(f"{image_list[i]} {image_list[j]}\n")


def run_mapper(colmap_bin: str, database: Path, image_dir: Path, sparse_dir: Path,
               n_images: int, mapper: str = "glomap"):
    if mapper == "colmap":
        print(f"[colmap] Running incremental mapper...", flush=True)
        cmd = [
            colmap_bin, "mapper",
            "--database_path", str(database),
            "--image_path",    str(image_dir),
            "--output_path",   str(sparse_dir),
        ]
        label = "colmap"
    else:
        print(f"[glomap] Running global_mapper with {colmap_bin}...", flush=True)
        cmd = [
            colmap_bin, "global_mapper",
            "--database_path", str(database),
            "--image_path",    str(image_dir),
            "--output_path",   str(sparse_dir),
        ]
        label = "glomap"

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Error: {label} mapper exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    if not (sparse_dir / "0").exists():
        print(f"Error: {label} produced no reconstruction in {sparse_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        recon = pycolmap.Reconstruction(str(sparse_dir / "0"))
        print(f"[{label}] Registered {len(recon.images)}/{n_images} images, {len(recon.points3D)} points",
              flush=True)
    except Exception:
        pass

    print(f"[{label}] Done → {sparse_dir}/0/", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--matcher", default="aliked+lightglue", choices=ALL_MATCHERS)
    ap.add_argument("--mapper", default="glomap", choices=["glomap", "colmap"],
                    help="glomap=global mapper (default), colmap=incremental mapper")
    ap.add_argument("--focal-length", type=float, default=None,
                    help="Prior focal length in pixels (fixes GLOMAP 'no prior' warning)")
    ap.add_argument("--colmap-bin", default=None,
                    help="Path to colmap binary with global_mapper (GLOMAP 4.x)")
    ap.add_argument("--sequential-overlap", type=int, default=10,
                    help="Frames to pair sequentially when >50 frames")
    args = ap.parse_args()

    image_dir: Path = args.image_dir
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    image_list = sorted(
        p.name for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    n = len(image_list)
    if n < 2:
        print(f"Error: need at least 2 images in {image_dir} (found {n})", file=sys.stderr)
        sys.exit(1)
    print(f"[hloc] {n} images in {image_dir}", flush=True)

    features_h5 = out / "features.h5"
    matches_h5  = out / "matches.h5"
    pairs_txt   = out / "pairs.txt"
    database    = out / "database.db"
    sparse_dir  = out / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    colmap_bin = args.colmap_bin
    if colmap_bin is None:
        import shutil
        colmap_bin = shutil.which("colmap") or "colmap"

    # ── Pair generation (same for both paths) ────────────────────────────────
    if n <= 50:
        print(f"[hloc] Generating exhaustive pairs ({n*(n-1)//2} pairs)...", flush=True)
        pairs_from_exhaustive.main(pairs_txt, image_list=image_list)
    else:
        print(f"[hloc] Generating sequential pairs (overlap={args.sequential_overlap})...", flush=True)
        make_sequential_pairs(image_list, args.sequential_overlap, pairs_txt)

    if args.matcher in SPARSE_FEATURE_CONFS:
        # ── Sparse path: extract features, then match ─────────────────────────
        feature_conf_name = SPARSE_FEATURE_CONFS[args.matcher]
        feature_conf = extract_features.confs[feature_conf_name]
        print(f"[hloc] Extracting {feature_conf_name} features...", flush=True)
        extract_features.main(feature_conf, image_dir, feature_path=features_h5)

        matcher_conf = match_features.confs[args.matcher]
        print(f"[hloc] Matching with {args.matcher}...", flush=True)
        match_features.main(matcher_conf, pairs_txt, features=features_h5, matches=matches_h5)

    else:
        # ── Semi-dense path: LoFTR simultaneously detects + matches ───────────
        dense_conf = DENSE_CONFS[args.matcher]
        print(f"[hloc] Semi-dense matching with {args.matcher}...", flush=True)
        match_dense.main(
            dense_conf, pairs_txt, image_dir,
            export_dir=out,
            features=features_h5,
            matches=matches_h5,
        )

    # ── Build COLMAP database ─────────────────────────────────────────────────
    if database.exists():
        database.unlink()
    print("[hloc] Building COLMAP database...", flush=True)
    create_empty_db(database)
    # Provide focal length prior when available — GLOMAP warns and degrades without it.
    # Prior is stored as camera params: for SIMPLE_PINHOLE model, params = [f, cx, cy].
    # We use PINHOLE (f, f, cx, cy) so both fx and fy are set.
    img_opts = {}
    if args.focal_length:
        # Sample one image to get image dimensions
        sample = next(image_dir.iterdir())
        try:
            import struct as _s
            with open(sample, "rb") as _f:
                _f.read(16)  # skip PNG header bytes
            from PIL import Image as _Img
            _w, _h = _Img.open(sample).size
        except Exception:
            _w = _h = 0
        img_opts = {
            "camera_model": "PINHOLE",
            "camera_params": f"{args.focal_length},{args.focal_length},{_w/2},{_h/2}",
        }
        print(f"[hloc] Using focal length prior: f={args.focal_length}px ({_w}×{_h})", flush=True)
    import_images(image_dir, database, pycolmap.CameraMode.SINGLE,
                  options=img_opts if img_opts else None)
    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features_h5)
        import_matches(image_ids, db, pairs_txt, matches_h5)

    # ── Geometric verification ────────────────────────────────────────────────
    print("[hloc] Geometric verification...", flush=True)
    estimation_and_geometric_verification(database, pairs_txt, verbose=False)

    # ── Reconstruction ────────────────────────────────────────────────────────
    run_mapper(colmap_bin, database, image_dir, sparse_dir, n, mapper=args.mapper)


if __name__ == "__main__":
    main()
