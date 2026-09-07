#!/usr/bin/env python3
"""
Extract training and SfM quality metrics from a completed pipeline run.

Usage:
    python extract_metrics.py <pod_dir> [<sparse_model_dir>]

Outputs a JSON object to stdout.  All values are floats or ints; missing
values are null so the consumer can always key-check without try/except.
"""

import json
import os
import struct
import sys
from glob import glob
from pathlib import Path


# ── TFEvents (nerfstudio splatfacto) ─────────────────────────────────────────

def _parse_tfevents(path: str) -> dict:
    """
    Return {tag: [(step, value), ...]} from a TensorBoard events file.
    Pure-Python proto decoder — no tensorflow dependency.
    """
    results: dict = {}
    with open(path, "rb") as f:
        raw = f.read()

    pos = 0
    while pos + 12 <= len(raw):
        try:
            length = struct.unpack_from("<Q", raw, pos)[0]
            pos += 12  # length + header crc
            if pos + length + 4 > len(raw):
                break
            record = raw[pos : pos + length]
            pos += length + 4  # data + crc

            step, summary_chunk = _decode_event(record)
            if summary_chunk is None:
                continue

            for tag, value in _decode_summary(summary_chunk):
                results.setdefault(tag, []).append((step, value))
        except Exception:
            pos += 1

    return results


def _read_varint(data: bytes, pos: int):
    v, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        v |= (b & 0x7F) << shift; shift += 7
        if not (b & 0x80):
            return v, pos


def _decode_event(record: bytes):
    """Return (step, summary_bytes) from a serialised Event proto."""
    step = 0; summary = None; pos = 0
    while pos < len(record):
        fb = record[pos]; pos += 1
        fn, wt = fb >> 3, fb & 7
        if wt == 0:
            v, pos = _read_varint(record, pos)
            if fn == 2: step = v
        elif wt == 1:
            pos += 8
        elif wt == 2:
            vl, pos = _read_varint(record, pos)
            chunk = record[pos : pos + vl]; pos += vl
            if fn == 5: summary = chunk
        elif wt == 5:
            pos += 4
        else:
            break
    return step, summary


def _decode_summary(data: bytes):
    """Yield (tag, float_value) pairs from a Summary proto."""
    pos = 0
    while pos < len(data):
        fb = data[pos]; pos += 1
        fn, wt = fb >> 3, fb & 7
        if wt != 2:
            if wt == 0:
                _, pos = _read_varint(data, pos)
            elif wt == 1: pos += 8
            elif wt == 5: pos += 4
            else: return
            continue
        vl, pos = _read_varint(data, pos)
        chunk = data[pos : pos + vl]; pos += vl
        if fn != 1: continue  # only Value entries

        tag = None; value = None; vp = 0
        while vp < len(chunk):
            vfb = chunk[vp]; vp += 1
            vfn, vwt = vfb >> 3, vfb & 7
            if vwt == 2:
                vvl, vp = _read_varint(chunk, vp)
                if vfn == 1:
                    tag = chunk[vp : vp + vvl].decode("utf-8", errors="ignore")
                vp += vvl
            elif vwt == 5:
                v = struct.unpack_from("<f", chunk, vp)[0]; vp += 4
                if vfn == 2: value = v
            elif vwt == 0:
                _, vp = _read_varint(chunk, vp)
            elif vwt == 1:
                vp += 8
            else:
                break

        if tag is not None and value is not None:
            yield tag, value


def extract_nerfstudio_metrics(pod_dir: str) -> dict:
    """Read PSNR/SSIM/LPIPS from nerfstudio TFEvents file."""
    tf_files = glob(os.path.join(pod_dir, "ns_train", "**", "events.out.tfevents*"), recursive=True)
    if not tf_files:
        return {}

    metrics = _parse_tfevents(tf_files[0])

    # "all images" tag is the holdout eval across the full set
    prefix = "Eval Images Metrics Dict (all images)/"
    result = {}
    for short, full_tag in [("psnr", prefix + "psnr"),
                             ("ssim", prefix + "ssim"),
                             ("lpips", prefix + "lpips")]:
        if full_tag in metrics and metrics[full_tag]:
            # Take the last recorded value (end of training)
            result[short] = round(float(metrics[full_tag][-1][1]), 4)

    # Fallback: single-image eval
    single_prefix = "Eval Images Metrics/"
    for short, full_tag in [("psnr", single_prefix + "psnr"),
                             ("ssim", single_prefix + "ssim"),
                             ("lpips", single_prefix + "lpips")]:
        if short not in result and full_tag in metrics and metrics[full_tag]:
            result[short] = round(float(metrics[full_tag][-1][1]), 4)

    return result


# ── gsplat simple_trainer log ─────────────────────────────────────────────────

def extract_gsplat_metrics(pod_dir: str) -> dict:
    """Read PSNR/SSIM/LPIPS + loss + Gaussian count from gsplat outputs."""
    result = {}

    # Val stats JSON written by simple_trainer eval() — prefer this for PSNR
    stats_dir = os.path.join(pod_dir, "gsplat_output", "stats")
    if os.path.isdir(stats_dir):
        val_files = sorted(glob(os.path.join(stats_dir, "val_step*.json")))
        if val_files:
            try:
                with open(val_files[-1]) as f:
                    s = json.load(f)
                for key in ("psnr", "ssim", "lpips"):
                    if key in s:
                        result[key] = round(float(s[key]), 4)
                if "num_GS" in s:
                    result["gaussian_count"] = int(s["num_GS"])
            except Exception:
                pass

    # Training log for loss + gaussian count (fallback if stats JSON missing)
    log = os.path.join(pod_dir, "02_train.log")
    if os.path.exists(log):
        import re
        with open(log) as f:
            for line in f:
                m = re.search(r"\[train\] step (\d+)/(\d+)\s+loss=([\d.]+)\s+GS=(\d+)", line)
                if m:
                    result["train_loss"] = round(float(m.group(3)), 5)
                    if "gaussian_count" not in result:
                        result["gaussian_count"] = int(m.group(4))

    return result


# ── SfM sparse model stats ────────────────────────────────────────────────────

def extract_sfm_metrics(sparse_dir: str) -> dict:
    """
    Extract registration stats from a COLMAP sparse model directory.
    Works with both text (.txt) and binary (.bin) formats.
    """
    if not sparse_dir or not os.path.isdir(sparse_dir):
        return {}

    result = {}

    # Registered image count from images.txt
    images_txt = os.path.join(sparse_dir, "images.txt")
    images_bin = os.path.join(sparse_dir, "images.bin")
    if os.path.exists(images_txt):
        count = 0
        with open(images_txt) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    count += 1
        result["registered_images"] = count // 2  # two lines per image
    elif os.path.exists(images_bin):
        try:
            with open(images_bin, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
            result["registered_images"] = n
        except Exception:
            pass

    # Point count from points3D.txt or .bin
    pts_txt = os.path.join(sparse_dir, "points3D.txt")
    pts_bin = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(pts_txt):
        count = sum(1 for l in open(pts_txt) if l.strip() and not l.startswith("#"))
        result["sfm_points"] = count
    elif os.path.exists(pts_bin):
        try:
            with open(pts_bin, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
            result["sfm_points"] = n
        except Exception:
            pass

    # Mean track length — how many views actually see the average point. This is the variable
    # that predicts whether the capture is view-graph starved, and `registered_images` hides
    # it completely (48/48 on the bad DJI job; 250/251 on 4f280a9e).
    #
    # It is also what decides whether exhaustive matching is worth paying for. Measured
    # 2026-09-07, same SfM, matcher as the only variable:
    #
    #   capture     baseline track len   exhaustive gain
    #   b36e3755    starved (6-8 pairs)      +0.78 dB
    #   4f280a9e    6.18                     +0.29 dB
    #   da329e40    7.54                     +0.02 dB  (for +269 s)
    #
    # i.e. the benefit tracks track length, NOT frame count — da329e40 is the LARGER capture
    # and gained nothing. Recording this is the prerequisite for escalating to exhaustive
    # conditionally instead of by a frame-count threshold.
    try:
        import pycolmap
        rec = pycolmap.Reconstruction(sparse_dir)
        if rec.num_points3D():
            result["track_len_mean"] = round(rec.compute_mean_track_length(), 3)
            result["obs_per_image_mean"] = round(
                rec.compute_mean_observations_per_reg_image(), 1)
    except Exception as e:
        # Absent metric, not a masked failure: every caller treats these as optional and the
        # reconstruction itself is unaffected. Say so rather than swallowing it silently.
        print(f"[metrics] track length unavailable ({type(e).__name__}: {e})", file=sys.stderr)

    return result


# ── Splat / Gaussian count ────────────────────────────────────────────────────

def extract_splat_stats(pod_dir: str) -> dict:
    """Count Gaussians from exported PLY or from training log."""
    # From gsplat log
    gsplat = extract_gsplat_metrics(pod_dir)
    if "gaussian_count" in gsplat:
        return {"gaussian_count": gsplat["gaussian_count"],
                "train_loss": gsplat.get("train_loss")}

    # From nerfstudio export log: "N Gaussians have NaN/Inf and M have low opacity, only export K/L"
    import re
    for logfile in ["02b_ns_export.log", "02_train.log"]:
        path = os.path.join(pod_dir, logfile)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                m = re.search(r"only export (\d+)/(\d+)", line)
                if m:
                    return {"gaussian_count": int(m.group(1))}
                # Alternative: "Total number of Gaussians: N"
                m2 = re.search(r"(?:Total|num).*?gaussians?.*?(\d+)", line, re.IGNORECASE)
                if m2:
                    return {"gaussian_count": int(m2.group(1))}

    return {}


# ── Brush log ────────────────────────────────────────────────────────────────

def extract_brush_metrics(pod_dir: str) -> dict:
    """Read PSNR/SSIM from Brush training log (RUST_LOG=brush_cli=info format).
    Brush emits: 'Eval iter N: PSNR X.XXXX, ssim Y.YYYY'
    We take the last eval line (end of training).
    """
    import re
    log = os.path.join(pod_dir, "02_train.log")
    if not os.path.exists(log):
        return {}
    result = {}
    pattern = re.compile(r"Eval iter \d+: PSNR ([\d.]+), ssim ([\d.]+)")
    with open(log) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                result["psnr"] = round(float(m.group(1)), 4)
                result["ssim"] = round(float(m.group(2)), 4)
    return result


# ── Reference-free splat/SfM health (no GPU/AI) ───────────────────────────────

def extract_ply_health(pod_dir: str, sparse_dir=None) -> dict:
    """Cheap reference-free quality signals persisted to pod.json so /history shows them:
    n_gaussians, floater %, giant-gaussian %, opacity, #SfM components, convergence.
    Wrapped so it can NEVER break the core metrics (returns partial/{} on any error)."""
    out: dict = {}
    try:
        import re as _re, glob as _g
        import numpy as np
        from plyfile import PlyData
        plys = sorted(_g.glob(os.path.join(pod_dir, "brush_output", "export_*.ply")),
                      key=lambda p: int(_re.findall(r"(\d+)", os.path.basename(p))[-1] or 0))
        if not plys:
            plys = sorted(_g.glob(os.path.join(pod_dir, "point_cloud", "iteration_*", "point_cloud.ply")))
        if plys:
            v = PlyData.read(plys[-1]).elements[0].data
            xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
            out["n_gaussians"] = int(len(xyz))
            if "scale_0" in v.dtype.names and "opacity" in v.dtype.names:
                scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
                opac = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
                ref = xyz
                if sparse_dir:
                    try:
                        import pycolmap
                        r = pycolmap.Reconstruction(sparse_dir)
                        if r.num_points3D() > 10:
                            ref = np.array([p.xyz for p in r.points3D.values()])
                    except Exception:
                        pass
                    try:
                        comps = [d for d in _g.glob(os.path.join(os.path.dirname(sparse_dir), "*"))
                                 if os.path.isdir(d) and os.path.basename(d).isdigit()]
                        out["sfm_components"] = len(comps)
                    except Exception:
                        pass
                lo, hi = np.percentile(ref, 1, 0), np.percentile(ref, 99, 0)
                ctr, half = (lo + hi) / 2, (hi - lo) / 2 + 1e-9
                extent = float(np.linalg.norm(hi - lo))
                maxs = scale.max(1)
                out["floater_frac_pct"] = round(100 * float(np.any(np.abs(xyz - ctr) > 1.5 * half, 1).mean()), 2)
                out["giant_gaussian_frac_pct"] = round(100 * float((maxs > 0.1 * extent).mean()), 3)
                out["opacity_p50"] = round(float(np.median(opac)), 3)
    except Exception:
        pass
    # convergence from brush eval trajectory
    try:
        import re as _re
        log = os.path.join(pod_dir, "02_train.log")
        if os.path.exists(log):
            ev = _re.findall(r"Eval iter (\d+): PSNR ([\d.]+)", open(log, errors="ignore").read())
            ev = [(int(i), float(p)) for i, p in ev]
            if len(ev) >= 2 and ev[-1][0] > ev[-4:][0][0]:
                tail = ev[-4:]
                slope = (tail[-1][1] - tail[0][1]) / ((tail[-1][0] - tail[0][0]) / 1000)
                out["psnr_slope_db_per_1k"] = round(slope, 3)
                out["converged"] = bool(abs(slope) < 0.15)
    except Exception:
        pass
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(json.dumps({}))
        return

    pod_dir = sys.argv[1]
    sparse_dir = sys.argv[2] if len(sys.argv) > 2 else None

    metrics = {}

    # Training metrics (nerfstudio, gsplat, or brush)
    ns_metrics = extract_nerfstudio_metrics(pod_dir)
    if ns_metrics:
        metrics.update(ns_metrics)
    else:
        gsplat = extract_gsplat_metrics(pod_dir)
        for key in ("psnr", "ssim", "lpips", "train_loss"):
            if key in gsplat:
                metrics[key] = gsplat[key]
        if "psnr" not in metrics:
            brush = extract_brush_metrics(pod_dir)
            metrics.update(brush)

    # Splat/Gaussian stats
    metrics.update(extract_splat_stats(pod_dir))

    # SfM stats
    if sparse_dir:
        sfm = extract_sfm_metrics(sparse_dir)
        metrics.update(sfm)

    # Reference-free splat/SfM health (persisted so /history shows beyond-PSNR quality)
    metrics.update(extract_ply_health(pod_dir, sparse_dir))

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
