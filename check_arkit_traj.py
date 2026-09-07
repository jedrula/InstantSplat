#!/usr/bin/env python3
"""Check an ARKit frames.traj against a prior-free reconstruction before trusting it as a prior.

WHY. `--sfm pose_prior` seeds COLMAP with the phone's poses at a deliberately tight 1 cm std.
That is the right call when the trajectory is sound and actively harmful when it is not, and
the two cases are indistinguishable without a check. Measured 2026-09-07 on two captures from
the same phone, same app, same session:

    capture     ARKit trajectory              pose_prior dPSNR   solved scale
    da329e40    clean                                  +0.30         0.9977
    4f280a9e    frames 21-38 drifted >1 m              -0.15         1.0139

On `4f280a9e` the damage is visible in the geometry, not just the score. Three independent
prior-free reconstructions (vocab_tree, exhaustive, COLMAP 4.2.0) all place frames 21-38 at
~123.6 cm from ARKit and agree with EACH OTHER there to 0.18 cm; a sim3 fitted on that block
alone collapses to 9.3 cm. So the block is internally right and merely displaced -- ARKit
drifted, the reconstructions did not. Feeding those poses in as a tight prior pulled the block
10 cm toward ARKit and degraded its local fit from 9.3 cm to 14.3 cm.

WHAT THIS MEASURES. Given a prior-free COLMAP model of the same frames:

  global residual   robust sim3 (RANSAC-trimmed, so one bad block cannot drag the fit) between
                    the model's camera centres and the ARKit positions
  local fit         a sim3 fitted on ONE window alone. This is the discriminator. A window that
                    is displaced but internally consistent -- ARKit's fault -- has a large
                    global residual and a SMALL local one. A window the reconstruction actually
                    got wrong is large in both.

A window is reported BAD when its global residual is far above the capture's own median and its
local fit is small: that is drift in the prior, and priors should not be trusted there.

Note the metric that does NOT work here: scoring a pose_prior model against ARKit is circular,
because that model was optimised toward ARKit. Always run this on a PRIOR-FREE model.

Usage:
  check_arkit_traj.py --traj pod/frames.traj --sparse pod/sparse/0 [--window 20] [--json out]
Exit status: 0 clean, 1 drift detected (so it can gate a pipeline), 2 could not check.
"""
import argparse
import json
import re
import sys

import numpy as np
import pycolmap


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(S)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    s = (D * d).sum() / (Xc ** 2).sum() * len(X)
    return s, R, my - s * R @ mx


def apply_sim3(s, R, t, X):
    return (s * (R @ X.T)).T + t


def robust_sim3(X, Y, thresh=0.25, iters=4):
    s, R, t = umeyama(X, Y)
    for _ in range(iters):
        keep = np.linalg.norm(apply_sim3(s, R, t, X) - Y, axis=1) < thresh
        if keep.sum() < 4:
            break
        s, R, t = umeyama(X[keep], Y[keep])
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--sparse", required=True,
                    help="a PRIOR-FREE COLMAP model of the same frames (glomap_sift etc). "
                         "Scoring a pose_prior model against ARKit is circular.")
    ap.add_argument("--window", type=int, default=20, help="frames per window")
    ap.add_argument("--drift-factor", type=float, default=4.0,
                    help="a window is suspect when its global residual exceeds this many times "
                         "the capture median")
    ap.add_argument("--local-ratio", type=float, default=0.5,
                    help="...AND its local fit is below this fraction of its global residual, "
                         "i.e. it is displaced but internally consistent -> the PRIOR is wrong")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    pos = np.array([[float(x) for x in l.split()[4:7]]
                    for l in open(a.traj) if len(l.split()) == 7])
    rec = pycolmap.Reconstruction(a.sparse)
    cen = {}
    for im in rec.images.values():
        m = re.search(r"(\d{4})\.(?:jpg|jpeg|png)$", im.name, re.I)
        if m:
            cen[int(m.group(1))] = im.projection_center()
    ids = np.array(sorted(i for i in cen if i < len(pos)))
    if len(ids) < 3 * a.window:
        print(f"[traj-check] only {len(ids)} frames matched; too few to window. SKIPPING.")
        return 2

    C = np.array([cen[i] for i in ids])
    A = pos[ids]
    s, R, t = robust_sim3(C, A)
    res = np.linalg.norm(apply_sim3(s, R, t, C) - A, axis=1)
    med = float(np.median(res))
    print(f"[traj-check] {len(ids)}/{len(pos)} frames, robust scale {s:.4f}, "
          f"median {med*100:.1f} cm, p90 {np.percentile(res,90)*100:.1f} cm, "
          f"max {res.max()*100:.1f} cm")

    bad, rows = [], []
    for lo in range(0, len(ids), a.window):
        sel = slice(lo, lo + a.window)
        wid, wC, wA, wr = ids[sel], C[sel], A[sel], res[sel]
        if len(wid) < 6:
            continue
        gl = float(np.median(wr))
        sl, Rl, tl = umeyama(wC, wA)
        loc = float(np.median(np.linalg.norm(apply_sim3(sl, Rl, tl, wC) - wA, axis=1)))
        suspect = gl > a.drift_factor * med and loc < a.local_ratio * gl
        rows.append({"frames": [int(wid[0]), int(wid[-1])], "n": len(wid),
                     "global_cm": gl * 100, "local_cm": loc * 100,
                     "local_scale": float(sl), "drift": bool(suspect)})
        if suspect:
            bad.append(rows[-1])

    print(f"[traj-check] {'frames':>12} {'global':>9} {'local fit':>10}  verdict")
    for r in rows:
        print(f"[traj-check] {str(r['frames'][0])+'-'+str(r['frames'][1]):>12} "
              f"{r['global_cm']:8.1f}c {r['local_cm']:9.1f}c  "
              f"{'ARKIT DRIFT' if r['drift'] else 'ok'}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"median_cm": med * 100, "scale": s, "windows": rows}, f, indent=2)

    if bad:
        span = ", ".join(f"{r['frames'][0]}-{r['frames'][1]}" for r in bad)
        print(f"[traj-check] DRIFT in {span}: displaced from the images but internally "
              f"consistent, so the TRAJECTORY is wrong there, not the reconstruction.")
        print(f"[traj-check] Priors from this traj will drag those frames onto bad poses. "
              f"Measured cost of doing so: -0.15 dB and local fit 9.3 -> 14.3 cm (4f280a9e).")
        return 1
    print("[traj-check] no drift windows — safe to use as a pose prior")
    return 0


if __name__ == "__main__":
    sys.exit(main())
