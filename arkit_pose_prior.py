#!/usr/bin/env python3
"""Reconstruct a capture with COLMAP's pose_prior_mapper, seeded by ARKit poses.

Invoked by `video_to_splat.sh --sfm pose_prior`. Emits a COLMAP model that the existing
`preposed_colmap` path then trains on, so this only has to produce sparse/0 + images.

WHY THIS EXISTS. Plain SfM has to bootstrap poses from matches alone, and that step is what
fails: on MuSHRoom's sauna it produced 44 cm LOCAL error, on honka a 30 cm global warp
(sub-cm locally). Both rooms are timber-lined — repetitive grain yields confident FALSE
matches. Seeding the mapper with the phone's visual-inertial poses turns pose estimation
from inference into refinement. Measured against those rooms' Faro laser scans:

    room    our plain SfM          pose priors      MuSHRoom's own SfM
    honka   30 cm warp             0.90 cm          0.91 cm
    sauna   broken (44 cm local)   2.62 cm          3.40 cm

i.e. it matches the reference pipeline on one room and beats it on the room where our SfM
broke outright.

NOT the same as freezing the poses and only triangulating: frozen noisy poses reject matches
at the reprojection test (867 points vs GLOMAP's 3,521 on the same 25 images). Priors let the
poses be refined by the images while still anchoring them.

PRIOR WEIGHT. Default 1 cm, deliberately tight. Swept 30/10/4/2/0.5 cm on two real ARKit
captures: a 10 m² capture sat FLAT at ~2.5-3.0 cm correction across the whole range (so the
prior never clamps a confident reconstruction — it is soft), while a weakly-constrained
2.6 m² close-range capture drifted 11.7 cm at 30 cm std and only settled at ~1.5 cm once the
prior was tight. Tight also produced the most points in both. Loose priors lose to
self-consistent false matches: sauna needed 0.5 cm (10 cm gave 22.57 cm vs Faro).

TRAJ CONVENTION. frames.traj holds ARKit's camera-to-world in ARKit's own basis, NOT the
ARKitScenes world-to-camera OpenCV convention its writer's comment claims. Verified twice:
aligning camera centres against GLOMAP (c2w 0.478 m vs w2c 1.994 m residual; the ARKit->OpenCV
basis flip separates 2.10 deg from 21.94 deg) and independently by triangulation point counts
(867 correct vs 54 without the flip, 60 read as w2c). Only POSITIONS are needed for a prior,
and those are the translation column directly — no flip required here.
"""
import argparse, glob, os, re, sqlite3, struct, subprocess, sys
import numpy as np
import cv2

ENV = dict(os.environ, QT_QPA_PLATFORM="offscreen")


def load_traj(path):
    """-> (N,3) camera positions, in capture order. Line i pairs with frame i."""
    pos, bad = [], 0
    for ln in open(path):
        t = ln.split()
        if len(t) != 7:
            bad += 1
            continue
        pos.append([float(x) for x in t[4:7]])
    return np.array(pos, dtype=np.float64), bad


def run(colmap, args, log_path):
    with open(log_path, "w") as lf:
        r = subprocess.run([colmap] + args, env=ENV, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"[pose_prior] {args[0]} failed rc={r.returncode} — see {log_path}")


def inject(db, positions_by_name, std):
    """Write camera positions into COLMAP's pose_priors table.

    Raw SQL because pycolmap 4.0.4 aborts in Database.open(). coordinate_system 1 =
    CARTESIAN (0 = WGS84, -1 = UNDEFINED); corr_sensor_type 0 = camera.
    """
    c = sqlite3.connect(db)
    rows = list(c.execute("SELECT image_id, name, camera_id FROM images"))
    if not rows:
        sys.exit("[pose_prior] database has no images — feature extraction produced nothing")
    cov = struct.pack("<9d", *(np.eye(3) * std ** 2).ravel())
    c.execute("DELETE FROM pose_priors")
    n = 0
    for image_id, name, camera_id in rows:
        key = os.path.splitext(os.path.basename(name))[0]
        if key not in positions_by_name:
            continue
        c.execute(
            "INSERT INTO pose_priors (pose_prior_id, corr_data_id, corr_sensor_id,"
            " corr_sensor_type, position, position_covariance, gravity, coordinate_system)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (image_id, image_id, camera_id, 0,
             struct.pack("<3d", *positions_by_name[key]), cov, None, 1))
        n += 1
    c.commit(); c.close()
    print(f"[pose_prior] injected {n}/{len(rows)} priors at {std*100:.2f} cm std")
    if n < len(rows):
        # Mixed primed/unprimed input is worse than either extreme: unprimed frames are
        # placed freely and drag the primed majority with them. Measured on sauna, where
        # 157 poseless frames pulled the main component to 76 cm.
        sys.exit(f"[pose_prior] {len(rows)-n} images have no prior. Unprimed frames drag the "
                 f"solution; exclude them from the input instead of mixing.")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--pincam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior-std", type=float, default=0.01, help="metres, 1-sigma")
    ap.add_argument("--matcher", default="", help="exhaustive|sequential|vocab_tree (auto)")
    ap.add_argument("--colmap", default=os.environ.get("COLMAP4_BIN", "colmap"),
                    help="MUST be COLMAP >= 4.0: 3.9.1 has no pose_prior_mapper, and the "
                         "two generations' SIFT descriptors are incompatible")
    a = ap.parse_args()

    names = sorted(n for n in os.listdir(a.images)
                   if n.lower().endswith((".jpg", ".jpeg", ".png")))
    pos, bad = load_traj(a.traj)
    if len(pos) != len(names):
        sys.exit(f"[pose_prior] frames.traj has {len(pos)} poses ({bad} malformed lines) but "
                 f"{len(names)} images. The traj is POSITIONAL — line i pairs with frame i — "
                 f"so a mismatch mispairs every pose. Refusing.")
    prior = {os.path.splitext(n)[0]: pos[i] for i, n in enumerate(names)}

    tok = open(a.pincam).read().split()
    w, h = int(float(tok[0])), int(float(tok[1]))
    fx, fy, cx, cy = (float(x) for x in tok[2:6])
    from PIL import Image
    iw, ih = Image.open(os.path.join(a.images, names[0])).size
    if (iw, ih) != (w, h):
        # Pre-f66e46e builds uploaded images upscaled by the screen scale while shipping the
        # original-resolution sidecar. The pair is still internally consistent once scaled.
        sx, sy = iw / w, ih / h
        print(f"[pose_prior] intrinsics say {w}x{h} but images are {iw}x{ih}; "
              f"rescaling by {sx:.4f}")
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
        w, h = iw, ih

    os.makedirs(a.out, exist_ok=True)
    db = os.path.join(a.out, "database.db")
    if os.path.exists(db):
        os.remove(db)

    # COLMAP prints its version banner through glog, i.e. on STDERR — reading only stdout
    # made the regex fail and reported "unknown", which then aborted a valid 4.0.4 install.
    _probe = subprocess.run([a.colmap, "point_triangulator", "--help"], env=ENV,
                            capture_output=True, text=True)
    banner = (_probe.stdout or "") + (_probe.stderr or "")
    m = re.search(r"COLMAP (\d+)\.(\d+)", banner)
    if not m or int(m.group(1)) < 4:
        sys.exit(f"[pose_prior] needs COLMAP >= 4.0, got '{m.group(0) if m else 'unknown'}' "
                 f"from {a.colmap}. Set COLMAP4_BIN.")
    print(f"[pose_prior] {len(names)} images, {m.group(0)}, {w}x{h} fx={fx:.1f}")

    # CPU SIFT: a 24.9 MP (5760x4320) frame OOMs SiftGPU on an 8 GB card, and the corrupted
    # CUDA context then fails EVERY subsequent image, leaving an empty database that only
    # surfaces two stages later. max_image_size alone does not prevent it.
    big = max(iw, ih) > 4000
    ext = ["--FeatureExtraction.max_image_size", "3200"]
    if big:
        ext += ["--FeatureExtraction.use_gpu", "0"]
        print("[pose_prior] oversized frames — extracting features on CPU")
    print("[pose_prior] 1/4 features")
    run(a.colmap, ["feature_extractor", "--database_path", db, "--image_path", a.images,
                   "--ImageReader.camera_model", "PINHOLE",
                   "--ImageReader.single_camera", "1",
                   "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}"] + ext,
        os.path.join(a.out, "01_features.log"))

    matcher = a.matcher or ("exhaustive" if len(names) <= 300 else "vocab_tree")
    print(f"[pose_prior] 2/4 matching ({matcher})")
    margs = [f"{matcher}_matcher", "--database_path", db]
    if matcher == "sequential":
        margs += ["--SequentialMatching.loop_detection", "1"]
    if matcher == "vocab_tree":
        vt = os.environ.get("VOCAB_TREE", "")
        if not os.path.exists(vt):
            sys.exit("[pose_prior] vocab_tree matcher needs VOCAB_TREE=/path/to/tree")
        margs += ["--VocabTreeMatching.vocab_tree_path", vt]
    run(a.colmap, margs, os.path.join(a.out, "02_match.log"))

    print("[pose_prior] 3/4 injecting priors")
    inject(db, prior, a.prior_std)

    print("[pose_prior] 4/4 pose_prior_mapper")
    sparse = os.path.join(a.out, "sparse")
    os.makedirs(sparse, exist_ok=True)
    run(a.colmap, ["pose_prior_mapper", "--database_path", db, "--image_path", a.images,
                   "--output_path", sparse], os.path.join(a.out, "03_map.log"))

    models = sorted(glob.glob(os.path.join(sparse, "*", "images.bin")))
    if not models:
        sys.exit(f"[pose_prior] no model produced — see {os.path.join(a.out,'03_map.log')}")

    def size(mb):
        with open(mb, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    # pose_prior_mapper can emit several components; keep the largest and say so, rather
    # than silently training on whichever COLMAP wrote to sparse/0.
    best = max(models, key=size)
    for mb in models:
        d = os.path.dirname(mb)
        with open(os.path.join(d, "points3D.bin"), "rb") as f:
            npts = struct.unpack("<Q", f.read(8))[0]
        mark = " <- using" if mb == best else ""
        print(f"[pose_prior]   model {os.path.basename(d)}: {size(mb)} images, {npts:,} points{mark}")
    if len(models) > 1:
        print(f"[pose_prior] NOTE {len(models)} components; only the largest is trained on")

    final = os.path.join(a.out, "sparse", "0")
    if os.path.dirname(best) != final:
        tmp = os.path.join(a.out, "sparse", "_chosen")
        os.rename(os.path.dirname(best), tmp)
        if os.path.exists(final):
            os.rename(final, os.path.join(a.out, "sparse", "_was0"))
        os.rename(tmp, final)
    link = os.path.join(a.out, "images")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(a.images), link)
    print(f"[pose_prior] -> {a.out}")


if __name__ == "__main__":
    main()
