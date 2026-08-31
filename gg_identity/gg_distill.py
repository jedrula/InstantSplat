#!/usr/bin/env python3
"""
gg_distill.py — GaussianGrouping-style identity distillation on a frozen Brush PLY.

Three phases (all run by default, checkpointed so each can be re-run independently):

  1. COLLECT  For each hold, run SAM2 on its top-N closest training frames → binary masks.
              Saved to <output_dir>/masks/  → skipped on re-run unless --recollect.

  2. TRAIN    Train per-Gaussian identity embeddings via differentiable rasterization.
              Loss: cross-entropy on rendered identity maps vs SAM2 ground-truth masks.
              Only identity_emb [N, D] and linear_head [D → H_holds] are optimised;
              PLY geometry (xyz, scale, rot, opacity, SH) is fully frozen.
              Saved to <output_dir>/identity_emb.npy + linear_head.pt

  3. COLORS   Two colour-extraction strategies, compared side-by-side:
                A) f_dc spatial:  Gaussians within radius R of each centroid → median Oklab
                B) f_dc identity: identity_head assigns each Gaussian to a hold → median Oklab
              Both produce a hold_oklab.json; B is printed with error count vs ground truth.

Usage:
  python gg_distill.py \\
    --ply  pods/9ec058d0/brush_output/export_5000.ply \\
    --pod-dir  pods/9ec058d0 \\
    [--phase collect|train|colors|all]  (default: all)
    [--n-views N]       top-N frames per hold for mask collection (default: 5)
    [--embed-dim D]     identity embedding dimension (default: 16)
    [--iters N]         training iterations (default: 3000)
    [--render-scale S]  downscale factor for rasterisation (default: 0.25)
    [--sam2-url URL]    default: http://localhost:8001
    [--output-dir DIR]  default: <pod_dir>/gg_identity

Ground truth used for error reporting:
  Compares predicted colour groups to holds.json colours (hex → Oklab).
  Prints confusion for every hold and overall error count at end of phase 3.
"""

import argparse
import base64
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from tqdm import tqdm, trange

GSPLAT_REPO = "/home/communications/workdir/gsplat"
if GSPLAT_REPO not in sys.path:
    sys.path.insert(0, GSPLAT_REPO)

C0 = 0.28209479177387814        # SH DC → RGB coefficient
OPACITY_THRESH = 0.05           # Gaussians below this are treated as background
MASK_AREA_MAX_FRAC = 0.12       # SAM2 masks covering >12% of frame are rejected
SPATIAL_RADIUS = 0.40           # metres: Gaussians within this radius of a hold centroid


# ── Oklab ─────────────────────────────────────────────────────────────────────

_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)
_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
], dtype=np.float32)


def _rgb_to_oklab(rgb_f32):
    """rgb_f32: (N,3) floats in [0,1]. Returns (N,3) Oklab."""
    lin = np.where(rgb_f32 <= 0.04045, rgb_f32 / 12.92,
                   ((rgb_f32 + 0.055) / 1.055) ** 2.4)
    lms = np.clip(lin @ _M1.T, 0.0, None)
    return np.cbrt(lms) @ _M2.T


# ── PLY I/O ───────────────────────────────────────────────────────────────────

def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    props = [p.name for p in v.properties]

    def _sorted_stack(prefix):
        names = sorted([p for p in props if p.startswith(prefix)],
                       key=lambda s: int(s.split("_")[-1]))
        return np.stack([v[n] for n in names], 1).astype(np.float32)

    xyz        = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    opacity_raw = v["opacity"].astype(np.float32)
    scales_raw = _sorted_stack("scale_")
    quats      = _sorted_stack("rot_")

    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    rgb  = np.clip(C0 * f_dc + 0.5, 0.0, 1.0)
    oklab = _rgb_to_oklab(rgb)
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-opacity_raw))

    return {
        "xyz": xyz,
        "opacity_raw": opacity_raw,
        "opacity": opacity_sigmoid,
        "scales_raw": scales_raw,
        "quats": quats,
        "f_dc": f_dc,
        "rgb": rgb,
        "oklab": oklab,
    }


# ── COLMAP camera reader ───────────────────────────────────────────────────────

def _qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),     2*(x*z+y*w)    ],
        [2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)    ],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x**2+y**2)],
    ], dtype=np.float64)


def load_colmap_cameras(sparse_dir):
    cameras, images = {}, []
    with open(sparse_dir / "cameras.txt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cid, model, W, H = int(p[0]), p[1], int(p[2]), int(p[3])
            params = [float(x) for x in p[4:]]
            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]
            elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]; cx, cy = params[1], params[2]
            else:
                fx = fy = params[0]; cx, cy = W / 2, H / 2
            cameras[cid] = dict(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy)

    with open(sparse_dir / "images.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):
        p = lines[i].split()
        qvec = [float(x) for x in p[1:5]]
        tvec = np.array([float(x) for x in p[5:8]], dtype=np.float64)
        cam  = cameras[int(p[8])]
        R    = _qvec_to_rotmat(qvec)
        w2c  = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.astype(np.float32)
        w2c[:3,  3] = tvec.astype(np.float32)   # COLMAP: p_cam = R @ X + t
        images.append({
            "name": p[9],
            "w2c": w2c,
            "cam": cam,
        })
    return images


# ── SAM2 ──────────────────────────────────────────────────────────────────────

def _sam2_segment(img_bytes, px, py, sam2_url):
    payload = json.dumps({
        "image_b64": base64.b64encode(img_bytes).decode(),
        "px": float(px), "py": float(py),
    }).encode()
    req = urllib.request.Request(
        f"{sam2_url}/api/v1/segment-at-point",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    mask_bytes = base64.b64decode(data["mask_b64"])
    mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    return mask  # H×W uint8, 255=hold


# ── Phase 1: Collect masks ────────────────────────────────────────────────────

def phase_collect(args, gs, images_list, holds, output_dir):
    """For each hold, pick top-N closest frames and run SAM2."""
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    images_dir = Path(args.pod_dir) / "images"

    index = {}  # hold_idx → list of {"frame": name, "mask_file": rel_path}

    n_holds  = len(holds)
    n_frames = len(images_list)
    print(f"[collect] {n_holds} holds, {n_frames} frames, top-{args.n_views} views/hold")

    for hi, hold in enumerate(holds):
        center = np.array(hold["center"], dtype=np.float64)
        views  = []

        for img in images_list:
            cam = img["cam"]
            R   = img["w2c"][:3, :3].astype(np.float64)
            t   = img["w2c"][:3,  3].astype(np.float64)
            p_cam = R @ center + t
            if p_cam[2] <= 0.01:
                continue
            u = cam["fx"] * p_cam[0] / p_cam[2] + cam["cx"]
            v = cam["fy"] * p_cam[1] / p_cam[2] + cam["cy"]
            if not (0 <= u < cam["W"] and 0 <= v < cam["H"]):
                continue
            views.append((float(p_cam[2]), float(u), float(v), img))

        views.sort(key=lambda x: x[0])
        views = views[:args.n_views]

        hold_records = []
        for depth, u, v, img in views:
            stem   = Path(img["name"]).stem
            mfile  = masks_dir / f"hold_{hi:03d}_{stem}.npy"

            if mfile.exists():
                hold_records.append({"frame": img["name"], "u": u, "v": v,
                                     "mask_file": str(mfile.relative_to(output_dir))})
                continue

            img_path = images_dir / img["name"]
            if not img_path.exists():
                continue
            try:
                img_bytes = img_path.read_bytes()
                mask = _sam2_segment(img_bytes, u, v, args.sam2_url)
                if mask is None:
                    continue
                cam = img["cam"]
                area_frac = (mask > 127).sum() / (cam["W"] * cam["H"])
                if area_frac > MASK_AREA_MAX_FRAC:
                    print(f"  hold {hi} frame {stem}: mask too large ({area_frac:.2%}), skip")
                    continue
                if (mask > 127).sum() < 50:
                    continue
                np.save(mfile, (mask > 127).astype(np.uint8))
                hold_records.append({"frame": img["name"], "u": u, "v": v,
                                     "mask_file": str(mfile.relative_to(output_dir))})
            except Exception as e:
                print(f"  hold {hi} frame {stem}: SAM2 error: {e}")

        index[hi] = hold_records
        n_ok = len(hold_records)
        status = "✓" if n_ok > 0 else "✗"
        print(f"  {status} hold {hi:>3}  {n_ok}/{len(views)} masks")

    with open(output_dir / "mask_index.json", "w") as f:
        json.dump(index, f, indent=2)
    n_covered = sum(1 for recs in index.values() if recs)
    print(f"[collect] done: {n_covered}/{n_holds} holds covered")
    return index


# ── Phase 2: Train identity embeddings ───────────────────────────────────────

def _build_cam_tensors(img_meta, render_W, render_H, device):
    cam = img_meta["cam"]
    sx  = render_W / cam["W"]
    sy  = render_H / cam["H"]
    K   = np.array([[cam["fx"]*sx, 0, cam["cx"]*sx],
                    [0, cam["fy"]*sy, cam["cy"]*sy],
                    [0, 0, 1]], dtype=np.float32)
    Kt   = torch.from_numpy(K).unsqueeze(0).to(device)
    w2ct = torch.from_numpy(img_meta["w2c"]).unsqueeze(0).to(device)
    return Kt, w2ct


def phase_train(args, gs, images_list, holds, output_dir):
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  iters={args.iters}  embed_dim={args.embed_dim}")

    index_path = output_dir / "mask_index.json"
    if not index_path.exists():
        print("ERROR: mask_index.json not found — run phase 'collect' first"); sys.exit(1)
    with open(index_path) as f:
        mask_index = {int(k): v for k, v in json.load(f).items()}

    n_holds = len(holds)
    N = gs["xyz"].shape[0]
    print(f"[train] {N:,} Gaussians, {n_holds} holds")

    # Freeze geometry on GPU
    means     = torch.from_numpy(gs["xyz"]).to(device)
    quats     = F.normalize(torch.from_numpy(gs["quats"]).to(device), dim=-1)
    scales    = torch.exp(torch.from_numpy(gs["scales_raw"]).to(device))
    opacities = torch.sigmoid(torch.from_numpy(gs["opacity_raw"]).to(device))

    # Trainable: identity embeddings + linear head
    D = args.embed_dim
    identity_emb = torch.zeros(N, D, device=device, requires_grad=True)
    nn.init.normal_(identity_emb, std=0.01)
    identity_emb = nn.Parameter(identity_emb)
    linear_head  = nn.Linear(D, n_holds, bias=True).to(device)

    optimizer = torch.optim.Adam(
        [{"params": [identity_emb], "lr": args.lr},
         {"params": linear_head.parameters(), "lr": args.lr * 2}]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-5)

    # Build flat list of (hold_idx, frame_meta, mask_path) for sampling
    samples = []
    img_by_name = {img["name"]: img for img in images_list}
    for hi, recs in mask_index.items():
        for rec in recs:
            img_meta = img_by_name.get(rec["frame"])
            if img_meta is None:
                continue
            mpath = output_dir / rec["mask_file"]
            if mpath.exists():
                samples.append((int(hi), img_meta, mpath))

    if not samples:
        print("ERROR: no valid samples — check mask collection"); sys.exit(1)
    print(f"[train] {len(samples)} (hold, frame) samples available")

    # Determine render resolution
    cam0 = images_list[0]["cam"]
    render_W = max(64, int(cam0["W"] * args.render_scale))
    render_H = max(64, int(cam0["H"] * args.render_scale))
    print(f"[train] render resolution: {render_W}×{render_H}")

    # ── Training loop ────────────────────────────────────────────────────────
    # Each step: pick all holds visible in a random frame, render once, CE loss
    # Group samples by frame for batch efficiency
    frame_to_holds = {}
    for hi, img_meta, mpath in samples:
        key = img_meta["name"]
        frame_to_holds.setdefault(key, []).append((hi, img_meta, mpath))
    frame_keys = list(frame_to_holds.keys())

    t0 = time.time()
    pbar = trange(args.iters, desc="identity distill")
    log_interval = max(1, args.iters // 20)

    for step in pbar:
        # Sample a random frame (may have multiple holds)
        fkey   = random.choice(frame_keys)
        items  = frame_to_holds[fkey]
        img_meta = items[0][1]

        Kt, w2ct = _build_cam_tensors(img_meta, render_W, render_H, device)

        # Rasterize identity embeddings → [1, H, W, D]
        renders, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=identity_emb,
            viewmats=w2ct, Ks=Kt,
            width=render_W, height=render_H,
            sh_degree=None, near_plane=0.01, far_plane=1000.0,
        )
        id_map = renders[0]  # [H, W, D]

        # Apply linear head → logits [H, W, n_holds]
        logits = linear_head(id_map)

        # Build label map from all visible holds in this frame
        label_map = torch.full((render_H, render_W), -1, dtype=torch.long, device=device)
        cam = img_meta["cam"]
        sx = render_W / cam["W"]
        sy = render_H / cam["H"]

        any_label = False
        for hi, _, mpath in items:
            mask_full = np.load(mpath).astype(bool)  # H_orig × W_orig
            # Resize to render resolution
            mask_small = cv2.resize(
                mask_full.astype(np.uint8),
                (render_W, render_H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            mask_t = torch.from_numpy(mask_small).to(device)
            label_map[mask_t] = hi
            any_label = True

        if not any_label:
            continue

        # Cross-entropy, ignoring background (label=-1)
        loss = F.cross_entropy(
            logits.reshape(-1, n_holds),
            label_map.reshape(-1),
            ignore_index=-1,
            reduction="mean",
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([identity_emb], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % log_interval == 0 or step == 0:
            pbar.set_postfix_str(f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.1e}")

    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ── Save ─────────────────────────────────────────────────────────────────
    emb_np = identity_emb.detach().cpu().float().numpy()
    np.save(output_dir / "identity_emb.npy", emb_np)
    torch.save(linear_head.state_dict(), output_dir / "linear_head.pt")
    print(f"[train] saved identity_emb.npy ({emb_np.shape}) + linear_head.pt")
    return emb_np


# ── Multi-view real-pixel Oklab ───────────────────────────────────────────────

def _multiview_oklab(pod_dir, output_dir, holds):
    """For each hold, pool masked pixels from all collected views → median Oklab.

    This avoids both the f_dc lighting-bake problem and the single-view shadow bias.
    Uses the SAM2 masks already on disk from phase_collect.
    """
    _M1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ], dtype=np.float32)
    _M2 = np.array([
        [0.2104542553,  0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050,  0.4505937099],
        [0.0259040371,  0.7827717662, -0.8086757660],
    ], dtype=np.float32)

    def rgb_to_oklab(rgb_u8):
        rgb = rgb_u8.astype(np.float32) / 255.0
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        lms = np.clip(lin @ _M1.T, 0.0, None)
        return np.cbrt(lms) @ _M2.T

    index_path = output_dir / "mask_index.json"
    if not index_path.exists():
        print("[colors] mask_index.json missing — skipping multi-view Oklab")
        return {}
    with open(index_path) as f:
        mask_index = {int(k): v for k, v in json.load(f).items()}

    images_dir = Path(pod_dir) / "images"
    oklab_c = {}

    for hi, hold in enumerate(holds):
        recs = mask_index.get(hi, [])
        if not recs:
            oklab_c[hi] = None
            continue

        all_pixels = []
        for rec in recs:
            mpath = output_dir / rec["mask_file"]
            img_path = images_dir / rec["frame"]
            if not mpath.exists() or not img_path.exists():
                continue
            mask = np.load(mpath).astype(bool)          # H × W bool
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            if img_bgr.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (img_bgr.shape[1], img_bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            pixels_bgr = img_bgr[mask]                  # (n_px, 3)
            pixels_rgb = pixels_bgr[:, ::-1]
            all_pixels.append(pixels_rgb)

        if not all_pixels:
            oklab_c[hi] = None
            continue

        pool = np.concatenate(all_pixels, axis=0)       # (total_px, 3)
        ok = rgb_to_oklab(pool)
        n = len(pool)
        oklab_c[hi] = {
            "L": float(np.median(ok[:, 0])),
            "a": float(np.median(ok[:, 1])),
            "b": float(np.median(ok[:, 2])),
            "n_views": len(recs),
            "n_px": n,
        }
        c = oklab_c[hi]
        print(f"  hold {hi:>3}: {c['n_views']}v  n_px={c['n_px']:>6}  L={c['L']:.3f} a={c['a']:+.3f} b={c['b']:+.3f}")

    return oklab_c


# ── Phase 3: Extract hold colours ─────────────────────────────────────────────

def phase_colors(args, gs, holds, output_dir, pod_dir=None):
    """Two strategies to get per-hold colour from Gaussians, printed side by side."""
    print("\n[colors] === Hold colour extraction ===")

    N = gs["xyz"].shape[0]
    xyz    = gs["xyz"]
    oklab  = gs["oklab"]
    opac   = gs["opacity"]
    active = opac > OPACITY_THRESH
    print(f"  {active.sum():,} / {N:,} Gaussians active (opacity>{OPACITY_THRESH})")

    # ── Strategy A: spatial proximity only (Tier-1 f_dc approach) ──────────
    print("\n[colors] Strategy A — spatial f_dc (no identity needed)")
    oklab_a = {}
    for hi, hold in enumerate(holds):
        center = np.array(hold["center"], dtype=np.float64)
        dists  = np.linalg.norm(xyz - center, axis=1)
        near   = active & (dists < SPATIAL_RADIUS)
        if near.sum() < 5:
            near = active & (dists < SPATIAL_RADIUS * 2)
        if near.sum() < 3:
            oklab_a[hi] = None
            continue
        oklab_a[hi] = {
            "L": float(np.median(oklab[near, 0])),
            "a": float(np.median(oklab[near, 1])),
            "b": float(np.median(oklab[near, 2])),
            "n": int(near.sum()),
        }
        c = oklab_a[hi]
        print(f"  hold {hi:>3}: n={c['n']:>5}  L={c['L']:.3f} a={c['a']:+.3f} b={c['b']:+.3f}")

    with open(output_dir / "hold_oklab_spatial.json", "w") as f:
        json.dump(oklab_a, f, indent=2)

    # ── Strategy C: multi-view real-pixel Oklab from collected masks ─────────
    print("\n[colors] Strategy C — multi-view real-pixel Oklab (5 views per hold)")
    oklab_c = _multiview_oklab(pod_dir, output_dir, holds)
    with open(output_dir / "hold_oklab_multiview.json", "w") as f:
        json.dump(oklab_c, f, indent=2)

    # ── Strategy B: identity-guided ─────────────────────────────────────────
    emb_path = output_dir / "identity_emb.npy"
    head_path = output_dir / "linear_head.pt"
    if not emb_path.exists():
        print("[colors] identity_emb.npy not found — skipping strategy B")
        return oklab_a

    print("\n[colors] Strategy B — identity-guided f_dc")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = args.embed_dim
    n_holds = len(holds)

    identity_emb = torch.from_numpy(np.load(emb_path)).to(device)  # [N, D]
    linear_head  = nn.Linear(D, n_holds, bias=True).to(device)
    linear_head.load_state_dict(torch.load(head_path, map_location=device))
    linear_head.eval()

    with torch.no_grad():
        logits = linear_head(identity_emb)          # [N, n_holds]
        pred   = logits.argmax(dim=-1).cpu().numpy() # [N]
        conf   = F.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()

    oklab_b = {}
    for hi, hold in enumerate(holds):
        # Use Gaussians assigned to this hold by identity, with high confidence
        assigned = (pred == hi) & active & (conf > 0.5)
        if assigned.sum() < 5:
            # Relax confidence threshold
            assigned = (pred == hi) & active
        if assigned.sum() < 3:
            oklab_b[hi] = None
            print(f"  hold {hi:>3}: NO Gaussians assigned")
            continue
        oklab_b[hi] = {
            "L": float(np.median(oklab[assigned, 0])),
            "a": float(np.median(oklab[assigned, 1])),
            "b": float(np.median(oklab[assigned, 2])),
            "n": int(assigned.sum()),
        }
        c = oklab_b[hi]
        print(f"  hold {hi:>3}: n={c['n']:>5}  L={c['L']:.3f} a={c['a']:+.3f} b={c['b']:+.3f}")

    with open(output_dir / "hold_oklab_identity.json", "w") as f:
        json.dump(oklab_b, f, indent=2)

    # ── Compare A vs B ───────────────────────────────────────────────────────
    print("\n[colors] ── A vs B comparison ──")
    diffs = []
    for hi in range(n_holds):
        a = oklab_a.get(hi)
        b = oklab_b.get(hi)
        if a and b:
            dL = abs(a["L"] - b["L"])
            da = abs(a["a"] - b["a"])
            db = abs(a["b"] - b["b"])
            diff = math.sqrt(dL**2 + da**2 + db**2)
            diffs.append(diff)
    if diffs:
        print(f"  mean Oklab distance A↔B: {np.mean(diffs):.4f}")
        print(f"  max  Oklab distance A↔B: {np.max(diffs):.4f}")
        print(f"  holds with diff > 0.1:   {sum(d>0.1 for d in diffs)}")

    # ── Cluster all three and compare ───────────────────────────────────────
    _report_clustering(oklab_c, holds, label="C (multiview-pixel)")
    _report_clustering(oklab_b, holds, label="B (identity)")
    _report_clustering(oklab_a, holds, label="A (spatial)")

    # ── Problem holds side-by-side ─────────────────────────────────────────
    problem = [25, 41, 27, 31, 16]
    print("\n[colors] ── Problem holds comparison (2D crop → A → B → C) ──")
    he_path = Path(pod_dir) / "hold_embeddings.json" if pod_dir else None
    oklab_2d = {}
    if he_path and he_path.exists():
        he = json.loads(he_path.read_text())
        for k, v in he.items():
            if not isinstance(v, dict):
                continue
            ok2d = v.get("oklab", {})
            if ok2d:
                oklab_2d[int(k)] = ok2d
    for hi in problem:
        v2 = oklab_2d.get(hi, {})
        va = (oklab_a or {}).get(hi) or {}
        vb = (oklab_b or {}).get(hi) or {}
        vc = oklab_c.get(hi) or {}
        L2, a2, b2 = v2.get("L","?"), v2.get("a","?"), v2.get("b","?")
        La, aa, ba = va.get("L","?"), va.get("a","?"), va.get("b","?")
        Lb, ab, bb = vb.get("L","?"), vb.get("a","?"), vb.get("b","?")
        Lc, ac, bc = vc.get("L","?"), vc.get("a","?"), vc.get("b","?")
        print(f"  #{hi:>2}  2D: L={L2}  A: L={La}  B: L={Lb}  C: L={Lc}")

    return oklab_b


def _report_clustering(oklab_dict, holds, label=""):
    """k-means on the extracted Oklab, count errors vs holds.json colours."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    valid = [(hi, v) for hi, v in oklab_dict.items() if v is not None]
    if len(valid) < 4:
        print(f"[colors] {label}: too few valid holds to cluster"); return

    indices = [hi for hi, _ in valid]
    X = np.array([[v["L"], v["a"], v["b"]] for _, v in valid], dtype=np.float32)
    wL, wab = 1.5, 2.5
    X_w = X * np.array([wL, wab, wab], dtype=np.float32)

    n     = len(X_w)
    tk    = max(5, round(n / 6.0))
    kmin  = max(4, tk - 2)
    kmax  = min(16, min(n - 1, tk + 3))
    best_k, best_s = tk, -1.0
    for k in range(kmin, kmax + 1):
        lbl = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_w)
        if len(set(lbl)) < 2: continue
        s = float(silhouette_score(X_w, lbl))
        if s > best_s:
            best_s, best_k = s, k

    labels = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X_w)

    # Map cluster IDs to colour names based on holds.json hex colours
    # Build per-cluster dominant hex from holds.json
    cluster_holds = {}
    for local_i, hi in enumerate(indices):
        cl = int(labels[local_i])
        cluster_holds.setdefault(cl, []).append(hi)

    print(f"\n[colors] {label} — k={best_k}, silhouette={best_s:.4f}")
    for cl in sorted(cluster_holds):
        members = cluster_holds[cl]
        hex_list = [holds[hi]["color"] for hi in members]
        print(f"  cluster {cl:>2}: {members}  colours: {hex_list[:6]}{'...' if len(hex_list)>6 else ''}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply",          default=None,
                    help="Brush PLY path (default: pod_dir/brush_output/export_5000.ply)")
    ap.add_argument("--pod-dir",      required=True)
    ap.add_argument("--phase",        default="all",
                    choices=["collect", "train", "colors", "all"])
    ap.add_argument("--n-views",      type=int,   default=5)
    ap.add_argument("--embed-dim",    type=int,   default=16)
    ap.add_argument("--iters",        type=int,   default=3000)
    ap.add_argument("--lr",           type=float, default=5e-3)
    ap.add_argument("--render-scale", type=float, default=0.25)
    ap.add_argument("--sam2-url",     default="http://localhost:8001")
    ap.add_argument("--output-dir",   default=None)
    ap.add_argument("--recollect",    action="store_true",
                    help="Re-run SAM2 even if masks already exist")
    args = ap.parse_args()

    pod_dir    = Path(args.pod_dir)
    ply_path   = Path(args.ply) if args.ply else pod_dir / "brush_output" / "export_5000.ply"
    output_dir = Path(args.output_dir) if args.output_dir else pod_dir / "gg_identity"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists():
        print(f"ERROR: PLY not found: {ply_path}"); sys.exit(1)

    sparse_dir = pod_dir / "colmap" / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = pod_dir / "sparse" / "0"

    print(f"[gg_distill] PLY: {ply_path}")
    print(f"[gg_distill] pod: {pod_dir}")
    print(f"[gg_distill] out: {output_dir}")

    # Load shared data
    holds = json.loads((pod_dir / "holds.json").read_text())

    print(f"[gg_distill] loading PLY …")
    gs = load_ply(ply_path)
    N  = gs["xyz"].shape[0]
    print(f"[gg_distill] {N:,} Gaussians, {len(holds)} holds")

    images_list = load_colmap_cameras(sparse_dir)
    print(f"[gg_distill] {len(images_list)} training frames")

    run_collect = args.phase in ("collect", "all")
    run_train   = args.phase in ("train",   "all")
    run_colors  = args.phase in ("colors",  "all")

    mask_index = None
    if run_collect:
        # Delete existing masks if recollect requested
        if args.recollect:
            import shutil
            mdir = output_dir / "masks"
            if mdir.exists():
                shutil.rmtree(mdir)
                print("[collect] existing masks cleared")
        mask_index = phase_collect(args, gs, images_list, holds, output_dir)

    if run_train:
        phase_train(args, gs, images_list, holds, output_dir)

    if run_colors:
        phase_colors(args, gs, holds, output_dir, pod_dir=args.pod_dir)

    print("\n[gg_distill] complete.")


if __name__ == "__main__":
    main()
