#!/usr/bin/env python3
"""
DeDoDe (decoupled Detect / Describe) sparse SfM front-end for GLOMAP/COLMAP.

Unlike LoFTR (detector-free / semi-dense), DeDoDe extracts ONE consistent
keypoint + descriptor set per image, so the same physical point keeps the same
ID across every pair — no per-pair keypoint reassignment loss. Detection and
description are two separate networks (3DV 2024), which places keypoints on
genuinely distinctive spots instead of every repetitive-texture edge.

Pipeline: DeDoDe detect+describe -> dual-softmax mutual-NN matching (kornia
match_smnn) -> hloc-format h5 -> COLMAP db + geometric verification -> GLOMAP.
Reuses the exact db-build / mapper code from glomap_hloc.py so the reconstruction
stage is identical to the other glomap_* SfM modes.

Usage:
  python glomap_dedode.py <image_dir> <output_dir>
       [--pairing exhaustive|sequential] [--sequential-overlap N]
       [--n-keypoints 10000] [--ratio-th 0.98]
       [--descriptor B|G] [--focal-length F]
       [--colmap-bin PATH] [--mapper glomap|colmap]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import pycolmap
from hloc.reconstruction import (
    create_empty_db,
    estimation_and_geometric_verification,
    get_image_ids,
    import_features,
    import_images,
    import_matches,
)
from hloc.utils.parsers import names_to_pair

# Reuse pair generation + mapper wrapper from the sibling hloc front-end.
from glomap_hloc import make_sequential_pairs, run_mapper


def make_exhaustive_pairs(image_list, out_path: Path):
    with open(out_path, "w") as f:
        n = len(image_list)
        for i in range(n):
            for j in range(i + 1, n):
                f.write(f"{image_list[i]} {image_list[j]}\n")


@torch.inference_mode()
def extract_all(model, image_dir: Path, image_list, n_kp: int, device):
    """Run DeDoDe on every image; return {name: (kp Nx2 float32, desc NxD f16)}."""
    import kornia as K

    feats = {}
    for idx, name in enumerate(image_list):
        img = K.io.load_image(str(image_dir / name), K.io.ImageLoadType.RGB32,
                              device=device)[None]
        kp, score, desc = model(img, n=n_kp)          # (1,N,2) (1,N) (1,N,D)
        feats[name] = (
            kp[0].float().cpu().numpy(),
            desc[0].half(),                            # keep on GPU for matching
        )
        if (idx + 1) % 25 == 0 or idx + 1 == len(image_list):
            print(f"[dedode] detect+describe {idx + 1}/{len(image_list)}", flush=True)
    return feats


def write_features_h5(feats, features_h5: Path, image_dir: Path):
    from PIL import Image
    with h5py.File(str(features_h5), "w") as f:
        for name, (kp, _desc) in feats.items():
            w, h = Image.open(image_dir / name).size
            grp = f.create_group(name)
            # import_features adds +0.5 (COLMAP origin); DeDoDe kps are already
            # pixel-centre, so subtract 0.5 to land back on DeDoDe coords.
            grp.create_dataset("keypoints", data=(kp - 0.5).astype(np.float32))
            grp.create_dataset("image_size", data=np.array([w, h]))


@torch.inference_mode()
def match_all(feats, pairs, matches_h5: Path, ratio_th: float):
    import kornia.feature as KF
    with h5py.File(str(matches_h5), "w") as f:
        for pi, (n0, n1) in enumerate(pairs):
            d0 = feats[n0][1]
            d1 = feats[n1][1]
            dists, idx = KF.match_smnn(d0.float(), d1.float(), th=ratio_th)
            N0 = d0.shape[0]
            matches0 = np.full(N0, -1, dtype=np.int32)
            scores0 = np.zeros(N0, dtype=np.float32)
            if idx.shape[0] > 0:
                i0 = idx[:, 0].cpu().numpy()
                i1 = idx[:, 1].cpu().numpy()
                matches0[i0] = i1
                scores0[i0] = (1.0 - dists[:, 0].cpu().numpy())  # smnn dist -> score
            grp = f.create_group(names_to_pair(n0, n1))
            grp.create_dataset("matches0", data=matches0)
            grp.create_dataset("matching_scores0", data=scores0)
            if (pi + 1) % 500 == 0 or pi + 1 == len(pairs):
                print(f"[dedode] match {pi + 1}/{len(pairs)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--pairing", choices=["exhaustive", "sequential"], default="auto",
                    nargs="?")
    ap.add_argument("--sequential-overlap", type=int, default=10)
    ap.add_argument("--n-keypoints", type=int, default=10000)
    ap.add_argument("--ratio-th", type=float, default=0.98,
                    help="Lowe ratio threshold for mutual-NN (higher = more matches)")
    ap.add_argument("--descriptor", choices=["B", "G"], default="B",
                    help="DeDoDe descriptor: B (fast) or G (best)")
    ap.add_argument("--focal-length", type=float, default=None)
    ap.add_argument("--colmap-bin", default=None)
    ap.add_argument("--mapper", default="glomap", choices=["glomap", "colmap"])
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    features_h5 = out / "features.h5"
    matches_h5 = out / "matches.h5"
    pairs_txt = out / "pairs.txt"
    database = out / "database.db"
    sparse_dir = out / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_list = sorted(p.name for p in args.image_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n = len(image_list)
    if n < 2:
        print(f"Error: need >=2 images, found {n}", file=sys.stderr)
        sys.exit(1)

    colmap_bin = args.colmap_bin
    if colmap_bin is None:
        import shutil
        colmap_bin = shutil.which("colmap") or "colmap"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Pairing ───────────────────────────────────────────────────────────────
    pairing = args.pairing
    if pairing in (None, "auto"):
        pairing = "exhaustive" if n <= 300 else "sequential"
    if pairing == "exhaustive":
        print(f"[dedode] exhaustive pairs ({n * (n - 1) // 2})...", flush=True)
        make_exhaustive_pairs(image_list, pairs_txt)
    else:
        print(f"[dedode] sequential pairs (overlap={args.sequential_overlap})...",
              flush=True)
        make_sequential_pairs(image_list, args.sequential_overlap, pairs_txt)
    pairs = [line.split() for line in pairs_txt.read_text().splitlines() if line.strip()]

    # ── DeDoDe detect + describe ──────────────────────────────────────────────
    import kornia.feature as KF
    print(f"[dedode] loading DeDoDe (detector L-upright, descriptor "
          f"{args.descriptor}-upright) on {device}...", flush=True)
    model = KF.DeDoDe.from_pretrained(
        detector_weights="L-upright",
        descriptor_weights=f"{args.descriptor}-upright",
    ).to(device).eval()
    feats = extract_all(model, args.image_dir, image_list, args.n_keypoints, device)

    print("[dedode] writing features.h5...", flush=True)
    write_features_h5(feats, features_h5, args.image_dir)

    print(f"[dedode] matching {len(pairs)} pairs (ratio_th={args.ratio_th})...",
          flush=True)
    match_all(feats, pairs, matches_h5, args.ratio_th)

    # ── Build COLMAP database (identical to glomap_hloc) ──────────────────────
    if database.exists():
        database.unlink()
    print("[dedode] building COLMAP database...", flush=True)
    create_empty_db(database)
    img_opts = {}
    if args.focal_length:
        from PIL import Image as _Img
        _w, _h = _Img.open(args.image_dir / image_list[0]).size
        img_opts = {
            "camera_model": "PINHOLE",
            "camera_params": f"{args.focal_length},{args.focal_length},{_w/2},{_h/2}",
        }
        print(f"[dedode] focal prior f={args.focal_length}px ({_w}×{_h})", flush=True)
    import_images(args.image_dir, database, pycolmap.CameraMode.SINGLE,
                  options=img_opts if img_opts else None)
    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, features_h5)
        import_matches(image_ids, db, pairs_txt, matches_h5)

    print("[dedode] geometric verification...", flush=True)
    estimation_and_geometric_verification(database, pairs_txt, verbose=False)

    run_mapper(colmap_bin, database, args.image_dir, sparse_dir, n, mapper=args.mapper)


if __name__ == "__main__":
    main()
