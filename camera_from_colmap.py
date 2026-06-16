#!/usr/bin/env python3
"""
Derive initial_camera.json for the splat viewer directly from the COLMAP sparse
reconstruction produced by any SfM step (mast3r/glomap/colmap/fastmap/etc.).

This is the trainer-agnostic replacement for extract_initial_camera.py.

The COLMAP sparse model (sparse/0/images.bin) has camera poses in COLMAP world
space.  The PLY/splat lives in a normalised version of that space.  We compute
the COLMAP → PLY transform per trainer from files they already write to disk:

  gsplat     : <result_dir>/colmap_to_ply_transform.npy  (saved by simple_trainer.py)
  splatfacto : <ns_process_dir>/transforms.json  +  <ns_train_dir>/**/dataparser_transforms.json
  pgsr       : <result_dir>/colmap_to_ply_transform.npy  (same approach as gsplat;
               you need to add the save call to pgsr/train.py if you want it)

If no transform file is found the output is in raw COLMAP world space — still
useful for inspecting relative orientation.

Usage
-----
  python camera_from_colmap.py \\
      --sparse   <path/to/sparse/0>     \\
      --trainer  gsplat|splatfacto|pgsr \\
      --result-dir <trainer output dir> \\
      --out      <path/to/initial_camera.json>

  # Pin a specific image by name (basename match):
      --frame frame_00040.png

  # For splatfacto, pass the nerfstudio *process* dir (contains transforms.json)
  # and the nerfstudio *train* dir (contains the dataparser_transforms.json subdir):
      --ns-process-dir <ns_input/>  --ns-train-dir <ns_train/>

Query-image workflow (future use)
----------------------------------
1. Register the query image against the existing sparse model with COLMAP:
     colmap image_registrator \\
         --database_path sparse/0/database.db \\
         --input_path sparse/0 \\
         --output_path /tmp/registered
2. Read the registered image's pose from /tmp/registered/images.bin.
3. Pass it through the same COLMAP → PLY transform produced here.
4. Feed position / look_at / up to the viewer.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

# Minimum wall height (meters in COLMAP space) visible when the scene loads.
# The initial camera is pulled back from the scene centroid until this many
# metres of wall are visible within the camera's vertical field of view.
MIN_WALL_HEIGHT_M = 3.0
# Additional metres added on top of the double-deficit pullback.
PULLBACK_EXTRA_M = 1.5


# ── COLMAP binary readers ──────────────────────────────────────────────────────

def _read_images_bin(path: Path):
    """Return list of dicts with keys: name, R (3×3), t (3,), image_id."""
    images = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz       = struct.unpack("<3d", f.read(24))
            camera_id        = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            # Skip 2D point observations
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * 24)   # each is (x float64, y float64, point3d_id int64)

            # quaternion (w,x,y,z) → rotation matrix  (world-to-camera)
            R = _quat_to_R(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            images.append({
                "image_id": image_id,
                "name": name.decode("utf-8"),
                "R": R,   # world-to-camera rotation
                "t": t,   # world-to-camera translation
            })
    return images


def _quat_to_R(qw, qx, qy, qz):
    """Unit quaternion → 3×3 rotation matrix."""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def colmap_c2w(img):
    """Convert COLMAP world-to-camera (R,t) → camera-to-world 4×4."""
    R, t = img["R"], img["t"]
    Rt = R.T
    pos = -Rt @ t
    c2w = np.eye(4)
    c2w[:3, :3] = Rt
    c2w[:3,  3] = pos
    return c2w   # OpenCV convention: X right, Y down, Z forward


# ── Transform loaders ─────────────────────────────────────────────────────────

# COLMAP uses OpenCV convention (Y down, Z forward).
# nerfstudio / our gsplat parser use OpenGL (Y up, Z backward).
# Flip Y and Z to convert COLMAP c2w → "pre-normalised" c2w.
_COLMAP_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def _load_transform_gsplat(result_dir: Path):
    """
    Load the 4×4 COLMAP-world → PLY transform saved by simple_trainer.py.
    Returns None if not found (fall back to identity with a warning).
    """
    p = result_dir / "colmap_to_ply_transform.npy"
    if not p.exists():
        print(f"Warning: {p} not found — output will be in COLMAP world space.", file=sys.stderr)
        return None
    T = np.load(str(p))   # (4,4), maps OpenGL-convention poses to training space
    return T


def _load_transform_splatfacto(ns_process_dir: Path, ns_train_dir: Path):
    """
    For splatfacto: transforms.json already has correct OpenGL-convention c2w poses
    in transforms.json world space; applied_transform (AT) maps them to ns_input space.
    ns-export gaussian-splat reverses dataparser_transforms when exporting, so the
    .splat lives in ns_input space.  We return R_dt/t_dt/scale for reference but
    run_from_transforms does NOT apply them.
    Returns (frames, AT_4x4, R_dt, t_dt, scale).
    """
    tf_path = ns_process_dir / "transforms.json"
    if not tf_path.exists():
        print(f"Warning: {tf_path} not found.", file=sys.stderr)
        return None
    d = json.loads(tf_path.read_text())
    at_raw = d.get("applied_transform")
    if at_raw:
        AT = np.vstack([at_raw, [0., 0., 0., 1.]])   # (4,4)
    else:
        AT = np.eye(4)

    dt_files = sorted(Path(ns_train_dir).rglob("dataparser_transforms.json")) if ns_train_dir else []
    if dt_files:
        dt = json.loads(dt_files[-1].read_text())
        T_raw = dt["transform"]   # 3×4
        scale  = dt["scale"]
        R_dt = np.array(T_raw)[:3, :3]
        t_dt = np.array(T_raw)[:3,  3]
        print(f"  dataparser_transforms: {dt_files[-1]}", flush=True)
        print(f"  dt_scale: {scale:.5f}", flush=True)
    else:
        print("Warning: dataparser_transforms.json not found — using ns_input space.", file=sys.stderr)
        R_dt = np.eye(3); t_dt = np.zeros(3); scale = 1.0

    frames = d.get("frames", [])
    return frames, AT, R_dt, t_dt, scale


# ── Scene centroid from splat ────────────────────────────────────────────────

def _scene_centroid_from_splat(splat_path: Path | None) -> np.ndarray | None:
    """
    Uniformly sample Gaussians from a binary .splat file and return their
    trimmed centroid in PLY/viewer world space.

    NOTE: ply2splat sorts by opacity descending, but nerfstudio Gaussians are
    nearly all fully opaque (alpha ≈ 255), so the sort order is arbitrary.
    Reading only the first N records gives a biased sample — we must sample
    uniformly across the file to get the true scene centroid.
    Returns None if the file is missing or unreadable.
    """
    if splat_path is None or not splat_path.exists():
        return None
    try:
        import struct as _struct
        record_bytes = 32
        with open(splat_path, "rb") as f:
            raw = f.read()
        n_total = len(raw) // record_bytes
        if n_total == 0:
            return None
        # Sample every K-th record for a representative, fast estimate
        step = max(1, n_total // 20_000)
        indices = range(0, n_total, step)
        pts = np.array([_struct.unpack_from("<3f", raw, i * record_bytes) for i in indices])
        # Trim 5-95th percentile to remove floaters
        p5, p95 = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
        mask = np.all((pts >= p5) & (pts <= p95), axis=1)
        centroid = pts[mask].mean(axis=0)
        print(f"  splat centroid ({len(pts)} samples, {mask.sum()} inliers): {centroid.round(4).tolist()}", flush=True)
        return centroid
    except Exception as e:
        print(f"Warning: could not read splat centroid from {splat_path}: {e}", file=sys.stderr)
        return None


# ── FOV + distance helpers ────────────────────────────────────────────────────

def _read_fov_y_from_cameras_txt(sparse_dir: Path) -> float | None:
    """Return vertical FOV in radians from sparse/0/cameras.txt, or None."""
    p = sparse_dir / "cameras.txt"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                model = parts[1].upper()
                try:
                    w, h = int(parts[2]), int(parts[3])
                except ValueError:
                    continue
                if not (0 < w < 20000 and 0 < h < 20000):
                    continue
                params = list(map(float, parts[4:]))
                # PINHOLE/OPENCV: params = [fx, fy, cx, cy, ...] → use fy
                if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV") and len(params) > 1:
                    fl_y = params[1]
                else:
                    fl_y = params[0]
                if fl_y > 0:
                    return 2 * math.atan(h / (2 * fl_y))
    except Exception:
        pass
    return None


def _pullback_position(pos: np.ndarray, look_at: np.ndarray,
                       fwd_level: np.ndarray, min_dist: float,
                       extra: float = 0.0) -> np.ndarray:
    """
    If pos is closer than min_dist to look_at (along fwd_level), pull it back.
    Applies the deficit twice then adds `extra` scene units on top.
    """
    current_depth = float(np.dot(look_at - pos, fwd_level))
    if current_depth < min_dist:
        target = 2 * min_dist - current_depth + extra
        new_pos = look_at - fwd_level * target
        print(f"  Pullback: depth {current_depth:.2f} → {target:.2f} units "
              f"(want {MIN_WALL_HEIGHT_M}m wall visible, +{extra:.2f} extra)", flush=True)
        return new_pos
    return pos


# ── Camera selection ──────────────────────────────────────────────────────────

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _select_best(cameras_training):
    """
    Pick the most "frontal" camera: central position, facing the scene centroid.
    `cameras_training` is a list of dicts with keys pos, fwd, up (all in training space).
    """
    positions = np.array([c["pos"] for c in cameras_training])
    centroid  = positions.mean(axis=0)
    median    = np.median(positions, axis=0)
    spread    = np.maximum(positions.max(axis=0) - positions.min(axis=0), 1e-3)

    def score(cam):
        to_c = _normalize(centroid - cam["pos"])
        if np.dot(cam["fwd"], to_c) < 0.2:
            return float("inf")
        err = np.abs(cam["pos"] - median) / spread
        # Weight: most spread axis last (usually depth), least weight
        w = np.ones(3)
        max_axis = int(np.argmax(spread))
        w[max_axis] = 0.5
        return float(np.dot(w, err))

    best = min(cameras_training, key=score)
    if score(best) == float("inf"):
        # All cameras face away — fall back to the one closest to centroid
        best = min(cameras_training, key=lambda c: np.linalg.norm(c["pos"] - centroid))
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def run_from_transforms(frames, AT, R_dt, t_dt, scale, out_path, pin_frame,
                        splat_path=None, fov_y: float | None = None):
    """
    For splatfacto: use the frames list from transforms.json directly.
    Poses are in OpenGL convention in transforms.json world space.

    CRITICAL: nerfstudio's ns-export gaussian-splat exports Gaussians in
    transforms.json world space (BEFORE applied_transform / AT).  The PLY header
    contains "Vertical Axis: z" (or y) set by nerfstudio to tell us which axis is
    physical-up in that raw export space.  AT is a nerfstudio-internal alignment
    that never affects the exported PLY — do NOT apply it here.

    The viewer gets positions + directions straight from transforms.json matrices,
    which are already in the same coordinate frame as the Gaussians.
    """
    cameras = []
    for fr in frames:
        M = np.array(fr["transform_matrix"])   # transforms.json world = PLY export space; NO AT
        pos = M[:3, 3]
        fwd = _normalize(-M[:3, 2])   # OpenGL: forward = -col2
        up  = _normalize( M[:3, 1])   # col1 = physical camera up (gravity-opposite)
        cameras.append({"name": fr["file_path"], "pos": pos, "fwd": fwd, "up": up})

    if pin_frame:
        pin_base = Path(pin_frame).name
        matched = [c for c in cameras if Path(c["name"]).name == pin_base]
        if not matched:
            print(f"Warning: --frame '{pin_frame}' not found; auto-selecting.", file=sys.stderr)
            best = _select_best(cameras)
        else:
            best = matched[0]
            print(f"  Pinned frame: {best['name']}", flush=True)
    else:
        best = _select_best(cameras)

    positions = np.array([c["pos"] for c in cameras])
    cam_centroid = positions.mean(axis=0)

    # look_at: mean forward across all cameras, levelled to be perpendicular to
    # physical up (so the view is horizontal, not tilted up/down).
    fwds = np.array([c["fwd"] for c in cameras])
    mean_fwd = _normalize(fwds.mean(axis=0))

    # World up = mean of all cameras' physical up direction (col1).
    ups = np.array([c["up"] for c in cameras])
    world_up = _normalize(ups.mean(axis=0))

    # Level the forward direction: remove component along world_up.
    fwd_level = mean_fwd - np.dot(mean_fwd, world_up) * world_up
    if np.linalg.norm(fwd_level) > 1e-6:
        fwd_level = _normalize(fwd_level)
    else:
        fwd_level = mean_fwd

    # Orbit target = scene center of gravity.
    # Prefer the centroid of the top-opacity Gaussians (they cluster on the wall
    # surface) over a blind look_dist estimate which is dominated by lateral
    # camera spread and overshoots by 5-6×.
    scene_centroid = _scene_centroid_from_splat(splat_path)
    if scene_centroid is not None:
        look_at = scene_centroid
    else:
        # Fallback: project from camera centroid along mean forward by a
        # fraction of the smallest non-lateral bbox dimension.
        bbox = positions.max(axis=0) - positions.min(axis=0)
        look_dist = float(np.sort(bbox)[1]) * 0.5   # median dimension * 0.5
        look_at = cam_centroid + fwd_level * look_dist

    # Ensure the initial camera is far enough back to see MIN_WALL_HEIGHT_M vertically.
    # splatfacto PLY is in ns_input space ≈ COLMAP metres, so min_dist is in metres.
    if fov_y and fov_y > 0:
        min_dist = (MIN_WALL_HEIGHT_M / 2) / math.tan(fov_y / 2)
    else:
        min_dist = 2.5
    initial_pos = _pullback_position(best["pos"], look_at, fwd_level, min_dist,
                                     extra=PULLBACK_EXTRA_M)

    result = {
        "position": [round(float(v), 4) for v in initial_pos],
        "look_at":  [round(float(v), 4) for v in look_at],
        "up":       [round(float(v), 4) for v in world_up],
        "source_frame": best["name"],
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"initial_camera.json → {out_path}")
    print(f"  frame:   {result['source_frame']}")
    print(f"  pos:     {result['position']}")
    print(f"  look_at: {result['look_at']}")
    print(f"  up:      {result['up']}")


def run(sparse_dir: Path, transform: np.ndarray | None,
        out_path: Path, pin_frame: str | None, splat_path: Path | None = None,
        native_colmap: bool = False):
    """
    Read images.bin from sparse_dir, transform poses, pick/pin a camera,
    write initial_camera.json.

    native_colmap=True: for trainers (e.g. Brush) whose output PLY lives in raw
    COLMAP world space.  No Y/Z flip is applied to positions; forward = +col2,
    up = -col1 (OpenCV convention: Y is down in the image).
    """
    images_bin = sparse_dir / "images.bin"
    if not images_bin.exists():
        print(f"Error: {images_bin} not found", file=sys.stderr)
        sys.exit(1)

    raw_images = _read_images_bin(images_bin)
    if not raw_images:
        print("Error: no images in images.bin", file=sys.stderr)
        sys.exit(1)

    # If no transform, use COLMAP→OpenGL flip (safe default for most trainers).
    if transform is None:
        T = _COLMAP_TO_OPENGL
    else:
        T = transform   # already incorporates flip_YZ for gsplat; for splatfacto too

    # Convert each image to training/PLY space
    cameras = []
    for img in raw_images:
        c2w_colmap = colmap_c2w(img)      # COLMAP OpenCV c2w

        if native_colmap:
            # Brush/COLMAP-native: Gaussians are in COLMAP world space.
            # Position needs no flip; forward = +col2 (+Z into scene); up = -col1.
            pos = c2w_colmap[:3, 3]
            fwd = _normalize( c2w_colmap[:3, 2])   # COLMAP: +Z is into scene
            up  = _normalize(-c2w_colmap[:3, 1])   # COLMAP: Y is down → negate
        else:
            c2w_training = T @ c2w_colmap           # in PLY/training space
            pos = c2w_training[:3, 3]
            # OpenGL convention: forward = -col2, physical up = col1
            fwd = _normalize(-c2w_training[:3, 2])
            up  = _normalize( c2w_training[:3, 1])

        cameras.append({"name": img["name"], "pos": pos, "fwd": fwd, "up": up})

    # Select frame
    if pin_frame:
        pin_base = Path(pin_frame).name
        matched = [c for c in cameras if Path(c["name"]).name == pin_base]
        if not matched:
            print(f"Warning: --frame '{pin_frame}' not found; auto-selecting.", file=sys.stderr)
            best = _select_best(cameras)
        else:
            best = matched[0]
            print(f"  Pinned frame: {best['name']}", flush=True)
    else:
        best = _select_best(cameras)

    positions = np.array([c["pos"] for c in cameras])
    centroid = positions.mean(axis=0)
    bbox_diag = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    look_dist = bbox_diag * 0.25

    # World up = mean camera col1 (physical up) in training space.
    # For gsplat: transform includes the full COLMAP→PLY chain so col1 of c2w_training
    # is the physical sensor up direction. Mean across cameras = gravity-opposite.
    fwds = np.array([c["fwd"] for c in cameras])
    ups  = np.array([c["up"]  for c in cameras])
    cam_centroid = np.array([c["pos"] for c in cameras]).mean(axis=0)
    mean_fwd = _normalize(fwds.mean(axis=0))
    world_up = _normalize(ups.mean(axis=0))

    # Level forward: remove world_up component so look_at is horizontal.
    fwd_level = mean_fwd - np.dot(mean_fwd, world_up) * world_up
    if np.linalg.norm(fwd_level) > 1e-6:
        fwd_level = _normalize(fwd_level)
    else:
        fwd_level = mean_fwd

    scene_centroid = _scene_centroid_from_splat(splat_path)
    if scene_centroid is not None:
        look_at = scene_centroid
    else:
        positions = np.array([c["pos"] for c in cameras])
        bbox = positions.max(axis=0) - positions.min(axis=0)
        look_dist = float(np.sort(bbox)[1]) * 0.5
        look_at = cam_centroid + fwd_level * look_dist

    # Ensure the initial camera is far enough back to see MIN_WALL_HEIGHT_M vertically.
    fov_y = _read_fov_y_from_cameras_txt(sparse_dir)
    if fov_y and fov_y > 0:
        min_dist_m = (MIN_WALL_HEIGHT_M / 2) / math.tan(fov_y / 2)
    else:
        min_dist_m = 2.5
    if native_colmap:
        # Brush: positions are in COLMAP world space ≈ metres.
        min_dist = min_dist_m
        extra = PULLBACK_EXTRA_M
    else:
        # Scale COLMAP metres → training units via the column norm of the applied transform.
        col_scale = float(np.linalg.norm(T[:3, 0]))
        scale = col_scale if col_scale > 1e-6 else 1.0
        min_dist = min_dist_m * scale
        extra = PULLBACK_EXTRA_M * scale
    initial_pos = _pullback_position(best["pos"], look_at, fwd_level, min_dist, extra=extra)

    result = {
        "position": [round(float(v), 4) for v in initial_pos],
        "look_at":  [round(float(v), 4) for v in look_at],
        "up":       [round(float(v), 4) for v in world_up],
        "source_frame": best["name"],
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(f"initial_camera.json → {out_path}")
    print(f"  frame:   {result['source_frame']}")
    print(f"  pos:     {result['position']}")
    print(f"  look_at: {result['look_at']}")
    print(f"  up:      {result['up']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparse",         required=True, help="Path to sparse/0 directory")
    ap.add_argument("--trainer",        default="gsplat",
                    choices=["gsplat", "splatfacto", "pgsr", "instantsplat", "brush"],
                    help="Which trainer produced the PLY")
    ap.add_argument("--result-dir",     help="Trainer output dir (gsplat/pgsr: contains colmap_to_ply_transform.npy)")
    ap.add_argument("--ns-process-dir", help="Nerfstudio process dir (splatfacto: contains transforms.json)")
    ap.add_argument("--ns-train-dir",   help="Nerfstudio train dir (splatfacto: parent of dataparser_transforms.json)")
    ap.add_argument("--frame",          help="Pin a specific image by filename (basename match)")
    ap.add_argument("--splat",          help="Path to binary .splat file (used to compute scene centroid as orbit target)")
    ap.add_argument("--out",            default="initial_camera.json", help="Output JSON path")
    args = ap.parse_args()

    sparse_dir = Path(args.sparse)
    out_path   = Path(args.out)

    # Brush: Gaussians live in raw COLMAP world space — no flip, native poses.
    if args.trainer == "brush":
        splat_path = Path(args.splat) if args.splat else None
        run(sparse_dir, None, out_path, args.frame, splat_path, native_colmap=True)
        return

    # Splatfacto: use transforms.json directly (correct image names + poses)
    if args.trainer == "splatfacto":
        ns_process = Path(args.ns_process_dir) if args.ns_process_dir else None
        ns_train   = Path(args.ns_train_dir)   if args.ns_train_dir   else None
        if not ns_process:
            print("Error: --ns-process-dir required for splatfacto", file=sys.stderr)
            sys.exit(1)
        result = _load_transform_splatfacto(ns_process, ns_train)
        if result is None:
            sys.exit(1)
        frames, AT, R_dt, t_dt, scale = result
        splat_path = Path(args.splat) if args.splat else None
        # Read FOV from transforms.json (top-level fl_y + h)
        fov_y_sf = None
        try:
            tf = json.loads((ns_process / "transforms.json").read_text())
            fl_y = tf.get("fl_y"); h = tf.get("h")
            if fl_y and h and fl_y > 0 and h > 0:
                fov_y_sf = 2 * math.atan(h / (2 * fl_y))
        except Exception:
            pass
        run_from_transforms(frames, AT, R_dt, t_dt, scale, out_path, args.frame,
                            splat_path, fov_y=fov_y_sf)
        return

    # All other trainers: read images.bin from sparse/0
    if args.trainer in ("gsplat", "pgsr", "instantsplat"):
        if not args.result_dir:
            print("Error: --result-dir required for gsplat/pgsr", file=sys.stderr)
            sys.exit(1)
        transform = _load_transform_gsplat(Path(args.result_dir))
        if transform is not None:
            transform = transform @ _COLMAP_TO_OPENGL
        else:
            transform = _COLMAP_TO_OPENGL
    else:
        transform = _COLMAP_TO_OPENGL

    splat_path = Path(args.splat) if args.splat else None
    run(sparse_dir, transform, out_path, args.frame, splat_path)


if __name__ == "__main__":
    main()
