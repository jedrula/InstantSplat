#!/usr/bin/env python3
"""SLAM-style loop verification of a COLMAP database against ARKit odometry, before the mapper runs.

A SLAM system accepts a loop closure only when it agrees with odometry. Here every verified pair that is not
a walk neighbour is a loop candidate, and ARKit is the odometry. Look-alike pairs (an identical facade on
another block) pass COLMAP's geometric verification -- 6529 of 42936 on f8e7c5c5 were >25 m apart -- and fold
GLOMAP. A candidate (i, j) is DELETED (from `matches` and `two_view_geometries`) when:
  * ARKit puts the cameras more than RADIUS m apart (no street-level camera pair overlaps that far), or
  * the relative pose the pair's own inlier matches imply (essential matrix -> R, t) disagrees with ARKit's:
    rotation by more than MAX_ROT deg, or -- when ARKit's baseline is long enough to have a direction --
    translation direction by more than MAX_DIR deg. A look-alike within RADIUS implies the cameras stand
    side by side in front of ONE wall; ARKit knows they did not.
Walk neighbours (|i - j| <= WINDOW in file-name order) are the odometry chain itself and are kept unchecked.
ARKit drifts metres over a long walk but its rotation and short-range geometry stay good, which is all this uses.
CPU only, seconds.

usage: loop_verify.py DATABASE FRAMES.TRAJ RADIUS_M [--max-rot 15] [--max-dir 30] [--window 10] [--min-baseline 0.5]
FRAMES.TRAJ is positional: line k is the k-th image by sorted name (ARKit: axis*angle cam-to-world, then position).
"""
import argparse, sqlite3, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('db'); ap.add_argument('traj'); ap.add_argument('radius', type=float)
ap.add_argument('--max-rot', type=float, default=15.0)
ap.add_argument('--max-dir', type=float, default=30.0)
ap.add_argument('--window', type=int, default=10)
ap.add_argument('--min-baseline', type=float, default=0.5)
a = ap.parse_args()
t0 = time.time()

rows = [l.split() for l in open(a.traj) if len(l.split()) == 7]
FLIP = np.diag([1.0, -1.0, -1.0])                     # ARKit camera (y up, -z forward) -> COLMAP (y down, +z forward)
def rot(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-12: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
C = np.array([[float(x) for x in r[4:7]] for r in rows])
Rc2w = [rot([float(x) for x in r[1:4]]) @ FLIP for r in rows]

db = sqlite3.connect(a.db)
names = dict(db.execute('select image_id, name from images'))
assert len(rows) == len(names), f'{a.traj}: {len(rows)} poses for {len(names)} images (positional)'
walk = {i: k for k, i in enumerate(sorted(names, key=lambda i: names[i]))}
cam_of = dict(db.execute('select image_id, camera_id from images'))
K = {}
for cid, model, w, h, params in db.execute('select camera_id, model, width, height, params from cameras'):
    p = np.frombuffer(params, np.float64)
    fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model in (0, 2, 3) else (p[0], p[1], p[2], p[3])   # SIMPLE_* vs PINHOLE/OPENCV
    K[cid] = (fx, fy, cx, cy)
kp_cache = {}
def kps(i):
    if i not in kp_cache:
        r, c, d = db.execute('select rows, cols, data from keypoints where image_id = ?', (i,)).fetchone()
        kp_cache[i] = np.frombuffer(d, np.float32).reshape(r, c)[:, :2]
    return kp_cache[i]
def norm(i, xy):
    fx, fy, cx, cy = K[cam_of[i]]
    return np.stack([(xy[:, 0] - cx) / fx, (xy[:, 1] - cy) / fy], 1).astype(np.float64)
ang = lambda R: np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

drop, why = [], {'far': 0, 'rot': 0, 'dir': 0, 'nopose': 0}
n_cand = n_all = 0
for pid, nr, data in db.execute('select pair_id, rows, data from two_view_geometries where rows > 0').fetchall():
    n_all += 1
    j = pid % 2147483647; i = (pid - j) // 2147483647
    wi, wj = walk[i], walk[j]
    if abs(wi - wj) <= a.window:
        continue
    n_cand += 1
    base = np.linalg.norm(C[wi] - C[wj])
    if base > a.radius:
        drop.append(pid); why['far'] += 1; continue
    m = np.frombuffer(data, np.uint32).reshape(nr, 2)
    x1, x2 = norm(i, kps(i)[m[:, 0]]), norm(j, kps(j)[m[:, 1]])
    E, inl = cv2.findEssentialMat(x1, x2, np.eye(3), cv2.RANSAC, 0.999, 1.0 / K[cam_of[i]][0])
    if E is None or E.shape != (3, 3):
        drop.append(pid); why['nopose'] += 1; continue
    _, R, t, _ = cv2.recoverPose(E, x1, x2, np.eye(3), mask=inl)
    R1, R2 = Rc2w[wi].T, Rc2w[wj].T                                   # world -> camera
    R_ark = R2 @ R1.T
    if ang(R @ R_ark.T) > a.max_rot:
        drop.append(pid); why['rot'] += 1; continue
    if base >= a.min_baseline:
        t_ark = R2 @ (C[wi] - C[wj]); t_ark /= np.linalg.norm(t_ark)
        if np.degrees(np.arccos(np.clip(float(t.ravel() @ t_ark), -1, 1))) > a.max_dir:
            drop.append(pid); why['dir'] += 1; continue
for table in ('matches', 'two_view_geometries'):
    db.executemany(f'delete from {table} where pair_id = ?', [(p,) for p in drop])
db.commit()
print(f'loop_verify: {n_all} verified pairs, {n_all - n_cand} walk neighbours kept, {n_cand} loop candidates checked; '
      f'deleted {len(drop)} (ARKit >{a.radius:g} m: {why["far"]}, rotation >{a.max_rot:g} deg: {why["rot"]}, '
      f'direction >{a.max_dir:g} deg: {why["dir"]}, no pose: {why["nopose"]}) in {time.time() - t0:.0f} s')
