#!/usr/bin/env python3
"""
localize_keyframe.py — Find camera pose of a query image within an existing
COLMAP sparse reconstruction. Outputs initial_camera.json compatible with
the 3D viewer.

Strategy (no database needed):
1. Parse cameras/images/points3D from sparse/0/
2. Build per-training-image KD-tree of 2D observations with known 3D points
3. Extract SIFT from training images + query; match query vs each training image
4. Look up 3D points via nearest-observation in training image (repeatable SIFT positions)
5. Collect 2D-3D correspondences; run OpenCV solvePnPRansac
6. Output position + look_at + up in COLMAP world coords

Usage:
  python localize_keyframe.py <sparse/0/> <images_dir/> <query.jpg> [output.json]
                              [--focal FL] [--min-inliers N]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import KDTree


# ── COLMAP parsing ────────────────────────────────────────────────────────────

def parse_cameras(sparse_dir: Path):
    """Returns {cam_id: {'w','h','fx','fy','cx','cy'}}"""
    cams = {}
    with open(sparse_dir / "cameras.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cam_id = int(p[0])
            model = p[1]
            w, h = int(p[2]), int(p[3])
            params = list(map(float, p[4:]))
            if model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:  # SIMPLE_RADIAL, RADIAL, etc.
                fl = params[0]
                fx = fy = fl
                cx, cy = params[1], params[2]
            cams[cam_id] = {"w": w, "h": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cams


def parse_images(sparse_dir: Path):
    """Returns {img_id: {'name', 'R', 't', 'cam_id', 'obs': array(N,3) [x,y,pt3d_id]}}"""
    imgs = {}
    with open(sparse_dir / "images.txt") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        header = lines[i].split()
        img_id = int(header[0])
        qw, qx, qy, qz = map(float, header[1:5])
        tx, ty, tz = map(float, header[5:8])
        cam_id = int(header[8])
        name = header[9]

        # quaternion → rotation matrix (world-to-camera)
        n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
        R = np.array([
            [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
            [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
        ])
        t = np.array([tx, ty, tz])

        # parse 2D observations
        obs_vals = list(map(float, lines[i+1].split()))
        obs = np.array(obs_vals).reshape(-1, 3)  # [x, y, point3d_id] per row

        imgs[img_id] = {"name": name, "R": R, "t": t, "cam_id": cam_id, "obs": obs}
        i += 2
    return imgs


def parse_points3d(sparse_dir: Path):
    """Returns {pt3d_id: np.array([X,Y,Z])}"""
    pts = {}
    with open(sparse_dir / "points3D.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            pts[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return pts


# ── Feature matching ──────────────────────────────────────────────────────────

def extract_sift(gray: np.ndarray, n_features: int = 8000):
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, descs = sift.detectAndCompute(gray, None)
    return kps, descs


def match_kps(desc_query, desc_train, ratio=0.75):
    """Lowe ratio test BF matching. Returns [(q_idx, t_idx), ...]."""
    if desc_query is None or desc_train is None or len(desc_train) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(desc_query.astype(np.float32),
                      desc_train.astype(np.float32), k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append((m.queryIdx, m.trainIdx))
    return good


# ── 2D-3D correspondence lookup ───────────────────────────────────────────────

def build_obs_lookup(obs: np.ndarray, pts3d: dict):
    """
    Build a KDTree over 2D observations that have a valid 3D point.
    Returns (kdtree, filtered_obs) where filtered_obs[i] = [x, y, pt3d_id].
    """
    valid = obs[obs[:, 2] > 0]  # filter out -1 (unmatched)
    # further filter: point must exist in pts3d
    valid = np.array([r for r in valid if int(r[2]) in pts3d])
    if len(valid) == 0:
        return None, None
    tree = KDTree(valid[:, :2])
    return tree, valid


def get_correspondences(query_kps, train_kps, matches, obs_tree, obs_valid, pts3d,
                        max_dist_px=3.0):
    """
    For each (q_idx, t_idx) match, find nearest 2D observation in training image.
    If within max_dist_px, record 2D-3D correspondence.
    Returns (pts_2d, pts_3d) arrays.
    """
    if obs_tree is None:
        return np.zeros((0, 2)), np.zeros((0, 3))

    pts2d, pts3d_list = [], []
    seen_pt3d = set()

    for q_idx, t_idx in matches:
        t_kp = train_kps[t_idx]
        tx, ty = t_kp.pt
        dist, nn_idx = obs_tree.query([tx, ty])
        if dist > max_dist_px:
            continue
        pt3d_id = int(obs_valid[nn_idx, 2])
        if pt3d_id in seen_pt3d or pt3d_id not in pts3d:
            continue
        seen_pt3d.add(pt3d_id)
        q_kp = query_kps[q_idx]
        pts2d.append(q_kp.pt)
        pts3d_list.append(pts3d[pt3d_id])

    if not pts2d:
        return np.zeros((0, 2)), np.zeros((0, 3))
    return np.array(pts2d, dtype=np.float64), np.array(pts3d_list, dtype=np.float64)


# ── PnP localization ──────────────────────────────────────────────────────────

def localize(pts2d, pts3d, K, min_inliers=12):
    """Run solvePnPRansac. Returns (R_c2w, t_world_pos) or None."""
    if len(pts2d) < min_inliers:
        return None, None, 0
    dist_coeffs = np.zeros(4)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist_coeffs,
        iterationsCount=2000,
        reprojectionError=8.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None:
        return None, None, 0
    n_inliers = len(inliers)
    # refine with inliers only
    R_mat, _ = cv2.Rodrigues(rvec)
    # world-to-camera: x_cam = R @ x_world + t
    # camera position in world: -R.T @ t
    pos = (-R_mat.T @ tvec).flatten()
    return R_mat, pos, n_inliers


# ── Camera output ─────────────────────────────────────────────────────────────

def make_initial_camera(R_w2c, pos_world, scene_center=None):
    """Build initial_camera.json dict from world-to-camera R and position."""
    # COLMAP: +Z into scene, +Y down in image
    fwd_world = R_w2c.T @ np.array([0.0, 0.0, 1.0])  # forward
    up_world  = R_w2c.T @ np.array([0.0, -1.0, 0.0]) # up (image Y is down)

    if scene_center is not None:
        # Project scene center onto the forward ray for a meaningful look_at
        to_center = np.array(scene_center) - pos_world
        look_dist = float(np.dot(to_center, fwd_world))
        look_dist = max(look_dist, 0.5)  # minimum 0.5m
    else:
        look_dist = 1.5
    look_at = pos_world + fwd_world * look_dist
    return {
        "position":     [round(float(v), 4) for v in pos_world],
        "look_at":      [round(float(v), 4) for v in look_at],
        "up":           [round(float(v), 4) for v in up_world],
        "source_frame": "query_localized",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sparse_dir")
    ap.add_argument("images_dir")
    ap.add_argument("query")
    ap.add_argument("output", nargs="?", default="localized_camera.json")
    ap.add_argument("--focal", type=float, default=None,
                    help="Override query focal length (pixels). Default: scale from training FOV.")
    ap.add_argument("--min-inliers", type=int, default=12,
                    help="Minimum PnP inliers to accept pose (default 12)")
    ap.add_argument("--ratio", type=float, default=0.75,
                    help="Lowe ratio for SIFT matching (default 0.75)")
    ap.add_argument("--max-obs-dist", type=float, default=3.0,
                    help="Max pixel distance for observation lookup (default 3.0)")
    ap.add_argument("--top-n", type=int, default=30,
                    help="Match against top-N training images by keypoint overlap (default 30)")
    args = ap.parse_args()

    sparse_dir  = Path(args.sparse_dir)
    images_dir  = Path(args.images_dir)
    query_path  = Path(args.query)
    output_path = Path(args.output)

    print("Loading reconstruction...")
    cams   = parse_cameras(sparse_dir)
    imgs   = parse_images(sparse_dir)
    pts3d  = parse_points3d(sparse_dir)
    print(f"  {len(cams)} cameras, {len(imgs)} images, {len(pts3d)} 3D points")

    # ── Determine query intrinsics ──────────────────────────────────────────
    query_img_bgr = cv2.imread(str(query_path))
    if query_img_bgr is None:
        # try loading RGBA with cv2 flags
        query_img_bgr = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
    if query_img_bgr is None:
        sys.exit(f"Cannot read query image: {query_path}")
    qh, qw = query_img_bgr.shape[:2]
    print(f"Query image: {qw}×{qh}")

    # training camera intrinsics (use first camera as reference)
    ref_cam = list(cams.values())[0]
    ref_fy = ref_cam["fy"]
    ref_h  = ref_cam["h"]

    if args.focal is not None:
        q_fl = args.focal
        print(f"  Using specified focal: {q_fl:.1f} px")
    else:
        # Scale reference focal length to query image height (same physical FOV_y)
        q_fl = ref_fy * (qh / ref_h)
        fov_y = 2 * math.degrees(math.atan(ref_h / (2 * ref_fy)))
        print(f"  Reference FOV_y: {fov_y:.1f}° → query fl: {q_fl:.1f} px (scaled {ref_fy:.1f} × {qh}/{ref_h})")

    K = np.array([
        [q_fl,   0.0, qw / 2.0],
        [0.0,   q_fl, qh / 2.0],
        [0.0,    0.0,      1.0],
    ])

    # ── Extract query features ──────────────────────────────────────────────
    print("Extracting query features...")
    query_gray = cv2.cvtColor(query_img_bgr, cv2.COLOR_BGR2GRAY)
    q_kps, q_descs = extract_sift(query_gray)
    print(f"  {len(q_kps)} keypoints")
    if q_descs is None or len(q_kps) < 20:
        sys.exit("Too few query keypoints — check image.")

    # ── Build observation KD-trees per training image ──────────────────────
    print("Building observation lookups...")
    obs_trees = {}
    for img_id, im in imgs.items():
        tree, valid_obs = build_obs_lookup(im["obs"], pts3d)
        obs_trees[img_id] = (tree, valid_obs)

    # ── Extract training features and accumulate 2D-3D correspondences ─────
    print(f"Matching query against {len(imgs)} training images...")
    all_pts2d = []
    all_pts3d = []
    match_stats = []
    # track raw SIFT matches for fallback nearest-camera selection
    raw_match_counts = {}

    training_imgs_sorted = sorted(imgs.items(), key=lambda x: x[0])
    for img_id, im in training_imgs_sorted:
        img_path = images_dir / im["name"]
        if not img_path.exists():
            continue
        train_bgr = cv2.imread(str(img_path))
        if train_bgr is None:
            continue
        train_gray = cv2.cvtColor(train_bgr, cv2.COLOR_BGR2GRAY)
        t_kps, t_descs = extract_sift(train_gray)
        if t_descs is None:
            continue

        matches = match_kps(q_descs, t_descs, ratio=args.ratio)
        raw_match_counts[img_id] = len(matches)

        # ── Homography filter to keep geometrically consistent matches ────
        if len(matches) >= 8:
            q_pts = np.float32([q_kps[m[0]].pt for m in matches])
            t_pts = np.float32([t_kps[m[1]].pt for m in matches])
            _, mask = cv2.findHomography(t_pts, q_pts, cv2.RANSAC, 6.0)
            if mask is not None:
                matches = [m for m, keep in zip(matches, mask.ravel()) if keep]

        tree, valid_obs = obs_trees[img_id]
        pts2d, pts3d_matched = get_correspondences(
            q_kps, t_kps, matches, tree, valid_obs, pts3d,
            max_dist_px=args.max_obs_dist
        )
        n_new = len(pts2d)
        match_stats.append((img_id, im["name"], len(matches), n_new))

        if n_new > 0:
            all_pts2d.append(pts2d)
            all_pts3d.append(pts3d_matched)

        if (len(match_stats) % 10) == 0:
            print(f"  [{len(match_stats)}/{len(imgs)}] processed, {sum(s[3] for s in match_stats)} correspondences so far")

    print(f"\nTop training images by 3D correspondences:")
    match_stats.sort(key=lambda x: -x[3])
    for img_id, name, n_matches, n_corr in match_stats[:10]:
        print(f"  {name}: {n_matches} matches → {n_corr} 3D corr")

    # ── Nearest-camera fallback ────────────────────────────────────────────
    # Best training image by raw SIFT match count (before homography filter)
    best_train_id = max(raw_match_counts, key=raw_match_counts.get)
    best_train_im = imgs[best_train_id]
    print(f"\nBest training image (raw matches): {best_train_im['name']} "
          f"({raw_match_counts[best_train_id]} matches)")
    # Compute that training camera's world position
    R_best = best_train_im["R"]
    t_best = best_train_im["t"]
    pos_nearest = -R_best.T @ t_best
    print(f"  Position: {pos_nearest.round(4)}")

    if not all_pts2d:
        sys.exit("No 2D-3D correspondences found — check image overlap.")

    pts2d_all = np.vstack(all_pts2d)
    pts3d_all = np.vstack(all_pts3d)
    print(f"\nTotal: {len(pts2d_all)} 2D-3D correspondences")

    # ── Deduplicate by closest 3D point (different training images may give ─
    #    the same pt3d_id via different training images) ─────────────────────
    seen = {}
    for i in range(len(pts2d_all)):
        # we don't track pt3d ids here — just keep all; PnP handles duplicates fine
        pass

    # ── PnP Ransac ──────────────────────────────────────────────────────────
    print("Running PnP RANSAC...")
    R_w2c, pos_world, n_inliers = localize(pts2d_all, pts3d_all, K,
                                            min_inliers=args.min_inliers)

    if R_w2c is None:
        print(f"PnP failed — fewer than {args.min_inliers} inliers.")
        print("  → Falling back to nearest training camera pose.")
        camera = make_initial_camera(best_train_im["R"], pos_nearest)
        camera["source_frame"] = f"nearest:{best_train_im['name']}"
        output_path.write_text(json.dumps(camera, indent=2))
        print(f"\nWrote {output_path} (nearest-camera fallback)")
        print(json.dumps(camera, indent=2))
        sys.exit(0)

    print(f"  PnP inliers: {n_inliers}/{len(pts2d_all)}")
    print(f"  Camera position: {pos_world.round(4)}")

    # ── Sanity check 1: position within reasonable scene extent ─────────────
    all_train_pos = np.array([-im["R"].T @ im["t"] for im in imgs.values()])
    env_min = all_train_pos.min(axis=0)
    env_max = all_train_pos.max(axis=0)
    env_center = (env_min + env_max) / 2
    env_radius = np.linalg.norm(env_max - env_min) / 2
    dist_from_center = np.linalg.norm(pos_world - env_center)

    # ── Sanity check 2: forward direction must roughly align with training ──
    all_train_fwd = np.array([im["R"][2] for im in imgs.values()])  # 3rd row of R_w2c = fwd in camera convention
    # In COLMAP: camera +Z is forward; fwd in world = R_w2c.T[:,2] = R_w2c[row2] transposed...
    # Actually: world_fwd = R_w2c.T @ [0,0,1] = R_w2c column 2
    all_train_fwd_world = np.array([im["R"].T[:, 2] for im in imgs.values()])
    mean_fwd = all_train_fwd_world.mean(axis=0)
    mean_fwd /= np.linalg.norm(mean_fwd)
    pnp_fwd = R_w2c.T[:, 2]  # localized forward direction
    pnp_fwd /= np.linalg.norm(pnp_fwd)
    fwd_dot = float(np.dot(pnp_fwd, mean_fwd))
    print(f"  Mean training fwd: {mean_fwd.round(3)}")
    print(f"  PnP fwd:           {pnp_fwd.round(3)}  (alignment dot: {fwd_dot:.2f})")

    use_nearest = False
    if dist_from_center > env_radius * 5:
        print(f"  WARNING: position {dist_from_center:.2f} from scene center "
              f"(envelope radius {env_radius:.2f}) — falling back to nearest camera.")
        use_nearest = True
    elif fwd_dot < 0.3:
        print(f"  WARNING: PnP forward direction misaligned with training cameras "
              f"(dot={fwd_dot:.2f}) — likely degenerate. Falling back to nearest camera.")
        use_nearest = True

    if use_nearest:
        R_w2c = best_train_im["R"]
        pos_world = pos_nearest
        n_inliers = 0

    # ── Try multiple focal lengths if inliers are borderline ─────────────────
    if n_inliers < 20 and args.focal is None:
        print("\nTrying alternative focal lengths to see if we can improve...")
        best_inliers = n_inliers
        best_result = (R_w2c, pos_world, n_inliers, q_fl)
        for test_fl in [q_fl * 0.6, q_fl * 0.75, q_fl * 1.25, q_fl * 1.5]:
            K_test = np.array([
                [test_fl,    0.0, qw / 2.0],
                [0.0,    test_fl, qh / 2.0],
                [0.0,        0.0,      1.0],
            ])
            R_t, p_t, n_t = localize(pts2d_all, pts3d_all, K_test,
                                      min_inliers=args.min_inliers)
            if R_t is not None and n_t > best_inliers:
                best_inliers = n_t
                best_result = (R_t, p_t, n_t, test_fl)
        R_w2c, pos_world, n_inliers, used_fl = best_result
        if used_fl != q_fl:
            print(f"  Better result with fl={used_fl:.1f}: {n_inliers} inliers")
            K = np.array([[used_fl, 0, qw/2], [0, used_fl, qh/2], [0, 0, 1]])
        else:
            print(f"  Original focal length {q_fl:.1f} still best ({n_inliers} inliers)")

    # ── Compute scene centroid (mean of 3D points) for look_at targeting ─────
    pts3d_arr = np.array(list(pts3d.values()))
    scene_center = pts3d_arr.mean(axis=0)
    print(f"  Scene centroid: {scene_center.round(3)}")

    # ── Output ───────────────────────────────────────────────────────────────
    camera = make_initial_camera(R_w2c, pos_world, scene_center=scene_center)
    if n_inliers == 0:
        camera["source_frame"] = f"nearest:{best_train_im['name']}"
    output_path.write_text(json.dumps(camera, indent=2))
    print(f"\nWrote {output_path}  (inliers: {n_inliers})")
    print(json.dumps(camera, indent=2))


if __name__ == "__main__":
    main()
