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
import argparse, glob, math, os, re, shutil, sqlite3, struct, subprocess, sys
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


def load_forward(path):
    """-> (N,3) unit viewing directions, same order as load_traj.

    The rotation columns were being thrown away, and with them the only thing that says two
    cameras cannot possibly see the same wall. Convention verified empirically rather than
    reasoned about, against 835 sim frames whose true headings are known: forward is
    R @ (0,0,-1) (ARKit's camera looks down -Z) and the world heading is atan2(fx, -fz),
    which reproduced every frame to 0.00 deg median AND 0.00 deg p90. Guessing this is how
    three earlier gravity/pose bugs happened; the check costs one script.
    """
    out = []
    for ln in open(path):
        t = ln.split()
        if len(t) != 7:
            continue
        R, _ = cv2.Rodrigues(np.array([float(t[1]), float(t[2]), float(t[3])]))
        f = R @ np.array([0.0, 0.0, -1.0])
        n = np.linalg.norm(f)
        out.append(f / n if n else np.array([0.0, 0.0, -1.0]))
    return np.array(out, dtype=np.float64)


def frustum_pairs(pos, fwd, names, max_dist, max_angle):
    """Pairs that could physically share a view: close enough, and not facing away.

    COLMAP's spatial_matcher pairs on POSITION only -- it was built for GPS, which carries no
    heading -- so it happily proposes two cameras standing back to back. Measured on
    block2ha's 9,729 verified baseline pairs, where 344 are strong (>=150 inliers) yet join
    cameras more than 60 m apart and are therefore impossible:

        angle between viewing directions    false pairs in band
          0- 60 deg                              0
         60- 90 deg                            170
         90-120 deg                            174
        120-180 deg                              0

    Every false constraint sits in the 60-120 deg band -- two cameras at right angles to each
    other, both photographing a building CORNER, and in a city where all 42 buildings share
    one facade texture every corner looks like every other corner. Rejecting above 90 deg
    removes 51% of them and costs ZERO of the 5,193 good strong pairs.
    """
    keep = []
    for i in range(len(pos)):
        d = np.linalg.norm(pos - pos[i], axis=1)
        c = fwd @ fwd[i]
        ok = (d <= max_dist) & (c >= math.cos(math.radians(max_angle)))
        for j in np.where(ok)[0]:
            if j > i:
                keep.append((names[i], names[int(j)]))
    return keep


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
    ap.add_argument("--matcher", default="",
                    help="exhaustive|sequential|vocab_tree|spatial|frustum (auto). 'spatial' "
                         "pairs by prior POSITION rather than appearance — the right choice "
                         "when a stretch of the capture is texture-poor. 'frustum' also uses "
                         "the recorded ORIENTATION, so two cameras standing close but facing "
                         "away from each other are never proposed")
    ap.add_argument("--frustum-max-angle", type=float, default=90.0,
                    help="degrees between viewing directions; frustum matcher only")
    ap.add_argument("--spatial-max-distance", type=float, default=2.5,
                    help="metres; spatial matcher only. Room-scale default, vs COLMAP's 100")
    ap.add_argument("--spatial-max-neighbors", type=int, default=40,
                    help="spatial matcher only")
    ap.add_argument("--verify-against", default="",
                    help="a PRIOR-FREE COLMAP model (e.g. the capture's glomap_sift sparse/0). "
                         "Runs check_arkit_traj.py first and REFUSES if the trajectory drifted: "
                         "a tight prior on a drifted traj measurably hurts (4f280a9e -0.15 dB, "
                         "local fit 9.3 -> 14.3 cm), while on a clean one it gains (da329e40 "
                         "+0.30 dB, +30%% points, scale 0.9977). Use --allow-drift to override.")
    ap.add_argument("--allow-drift", action="store_true",
                    help="proceed even if --verify-against reports drift")
    ap.add_argument("--components", choices=("merge", "largest"), default="merge",
                    help="what to do when pose_prior_mapper emits several components. "
                         "'merge' concatenates them — valid here because they all share the "
                         "ARKit metric frame, and verified per component before merging. "
                         "'largest' is the old behaviour and throws the rest away")
    ap.add_argument("--colmap", default=os.environ.get("COLMAP_BIN", "colmap"),
                    help="MUST be COLMAP >= 4.0: 3.9.1 has no pose_prior_mapper, and the "
                         "two generations' SIFT descriptors are incompatible")
    ap.add_argument("--camera-model", choices=("OPENCV", "PINHOLE"),
                    default=os.environ.get("POSE_PRIOR_CAMERA_MODEL", "OPENCV"),
                    help="OPENCV (default): the ARKit intrinsics plus k1,k2,p1,p2 starting at 0, "
                         "refined by the mapper, then the model and images are undistorted to "
                         "PINHOLE so downstream sees the same layout as before. ARKit's intrinsics "
                         "carry no distortion, but the frames have it — Spirula's own SfM fitted "
                         "k1=0.055 k2=-0.066 on ab29e999. PINHOLE is the old behaviour")
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
                 f"from {a.colmap}. Set COLMAP_BIN.")
    print(f"[pose_prior] {len(names)} images, {m.group(0)}, {w}x{h} fx={fx:.1f}")

    # Gate on trajectory quality BEFORE spending the GPU. ARKit poses are only worth seeding
    # with when the trajectory is sound, and the two cases look identical from here without a
    # prior-free reconstruction to compare against.
    if a.verify_against:
        rc = subprocess.run([sys.executable,
                             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "check_arkit_traj.py"),
                             "--traj", a.traj, "--sparse", a.verify_against]).returncode
        if rc == 1 and not a.allow_drift:
            sys.exit("[pose_prior] refusing: the ARKit trajectory drifted where the images do "
                     "not agree, so a tight prior would pull the reconstruction onto bad poses. "
                     "Re-run with --allow-drift to override, or use --sfm glomap_sift.")
        if rc == 2:
            print("[pose_prior] trajectory check could not run; proceeding unverified")

    # CPU SIFT: a 24.9 MP (5760x4320) frame OOMs SiftGPU on an 8 GB card, and the corrupted
    # CUDA context then fails EVERY subsequent image, leaving an empty database that only
    # surfaces two stages later. max_image_size alone does not prevent it.
    big = max(iw, ih) > 4000
    ext = ["--FeatureExtraction.max_image_size", "3200"]
    if big:
        ext += ["--FeatureExtraction.use_gpu", "0"]
        print("[pose_prior] oversized frames — extracting features on CPU")
    print("[pose_prior] 1/4 features")
    cam_params = f"{fx},{fy},{cx},{cy}" + (",0,0,0,0" if a.camera_model == "OPENCV" else "")
    run(a.colmap, ["feature_extractor", "--database_path", db, "--image_path", a.images,
                   "--ImageReader.camera_model", a.camera_model,
                   "--ImageReader.single_camera", "1",
                   "--ImageReader.camera_params", cam_params] + ext,
        os.path.join(a.out, "01_features.log"))

    matcher = a.matcher or ("exhaustive" if len(names) <= 300 else "vocab_tree")

    # Priors go in BEFORE matching, not after: the spatial matcher SELECTS PAIRS from the
    # pose_priors table, so for that matcher an empty table means no pairs at all. Harmless
    # for the others — the mapper reads the same rows either way.
    print("[pose_prior] 2/4 injecting priors")
    inject(db, prior, a.prior_std)

    print(f"[pose_prior] 3/4 matching ({matcher})")
    if matcher == "frustum":
        fwd = load_forward(a.traj)
        if len(fwd) < len(names):
            sys.exit(f"[pose_prior] frustum matcher needs a rotation per image; "
                     f"traj has {len(fwd)} usable rows for {len(names)} images")
        pos_by_name = np.array([prior[os.path.splitext(n)[0]] for n in names])
        fwd_by_name = fwd[:len(names)]
        pairs = frustum_pairs(pos_by_name, fwd_by_name, names,
                              a.spatial_max_distance, a.frustum_max_angle)
        if not pairs:
            sys.exit("[pose_prior] frustum matcher produced no pairs — check the radius")
        plist = os.path.join(a.out, "frustum_pairs.txt")
        with open(plist, "w") as fh:
            for u, v in pairs:
                fh.write(f"{u} {v}\n")
        print(f"[pose_prior]   {len(pairs)} pairs within {a.spatial_max_distance} m "
              f"and {a.frustum_max_angle} deg -> {plist}")
        run(a.colmap, ["matches_importer", "--database_path", db,
                       "--match_list_path", plist, "--match_type", "pairs",
                       "--FeatureMatching.use_gpu", "1"],
            os.path.join(a.out, "02_match.log"))
        matcher_done = True
    else:
        matcher_done = False
    margs = [f"{matcher}_matcher", "--database_path", db]
    if matcher == "spatial":
        # Pair by WHERE THE CAMERA WAS instead of by what the image looks like. This is the
        # direct answer to b36e3755: appearance retrieval starved on blank white walls
        # (frames 250-449 got 6-8 pairs each vs 31-58 elsewhere) even though the phone knew
        # its position the whole time.
        #
        # Two defaults are actively wrong for an indoor capture:
        #   ignore_z=1    built for aerial GPS, where z is altitude. ARKit is Y-up, so
        #                 COLMAP's "z" is one of our HORIZONTAL axes — ignoring it collapses
        #                 the flat onto a plane and calls rooms 6 m apart neighbours.
        #   max_distance=100 (metres) — larger than any flat, i.e. no pruning at all.
        margs += ["--SpatialMatching.ignore_z", "0",
                  "--SpatialMatching.max_distance", str(a.spatial_max_distance),
                  "--SpatialMatching.max_num_neighbors", str(a.spatial_max_neighbors)]
    if matcher in ("sequential", "vocab_tree"):
        # Both need the tree: vocab_tree to retrieve at all, and sequential because COLMAP's
        # loop_detection is itself vocab-tree retrieval and its option check aborts without a
        # readable path. Sequential+loop is the right default for a walkthrough — it pairs
        # temporal neighbours BY CONSTRUCTION rather than by appearance, which is what
        # vocab_tree alone failed to do on b36e3755's low-texture stretch.
        vt = os.environ.get("VOCAB_TREE", "")
        if not os.path.exists(vt):
            sys.exit(f"[pose_prior] {matcher} matcher needs VOCAB_TREE=/path/to/tree")
    if matcher == "sequential":
        margs += ["--SequentialMatching.loop_detection", "1",
                  "--SequentialMatching.vocab_tree_path", vt]
    if matcher == "vocab_tree":
        margs += ["--VocabTreeMatching.vocab_tree_path", vt]
    if not matcher_done:
        run(a.colmap, margs, os.path.join(a.out, "02_match.log"))

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
    for mb in models:
        d = os.path.dirname(mb)
        with open(os.path.join(d, "points3D.bin"), "rb") as f:
            npts = struct.unpack("<Q", f.read(8))[0]
        print(f"[pose_prior]   model {os.path.basename(d)}: {size(mb)} images, {npts:,} points")

    final = os.path.join(a.out, "sparse", "0")
    if len(models) > 1 and a.components == "merge":
        # MERGE, don't discard. pose_prior_mapper splits routinely — b36e3755 run D gave 7
        # components and keeping the largest trained on 240 of 515 frames. Every component is
        # anchored to the SAME ARKit metric frame, so unlike ordinary SfM components they are
        # already mutually registered and can be concatenated. Merging that run reached
        # 439/515 frames with median placement error 16.5 cm -> 9.4 cm and scale 1.0006.
        # merge_pose_prior_components.py re-verifies the shared frame per component and
        # ABORTS if any disagrees, so a genuinely misaligned component cannot slip through.
        print(f"[pose_prior] {len(models)} components — merging (--components largest to opt out)")
        merged = os.path.join(a.out, "sparse", "_merged")
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "merge_pose_prior_components.py"),
                        "--sparse", sparse, "--out", merged, "--traj", a.traj],
                       check=True)
        for d in glob.glob(os.path.join(sparse, "[0-9]*")):
            if os.path.isdir(d):
                os.rename(d, os.path.join(sparse, "_unmerged_" + os.path.basename(d)))
        os.rename(merged, final)
    else:
        # Largest component only. Kept as an escape hatch and for single-component models.
        best = max(models, key=size)
        if len(models) > 1:
            print(f"[pose_prior] NOTE {len(models)} components; only the largest is trained on")
        if os.path.dirname(best) != final:
            tmp = os.path.join(a.out, "sparse", "_chosen")
            os.rename(os.path.dirname(best), tmp)
            if os.path.exists(final):
                os.rename(final, os.path.join(a.out, "sparse", "_was0"))
            os.rename(tmp, final)
    link = os.path.join(a.out, "images")
    if a.camera_model != "PINHOLE":
        # Downstream (preposed_colmap, every trainer) expects a PINHOLE sparse/0 next to images/.
        # Undistort here so that contract holds: the fitted model is kept as sparse/0_distorted.
        und = os.path.join(a.out, "undistorted")
        shutil.rmtree(und, ignore_errors=True)
        run(a.colmap, ["image_undistorter", "--image_path", a.images, "--input_path", final,
                       "--output_path", und, "--output_type", "COLMAP"],
            os.path.join(a.out, "04_undistort.log"))
        if not os.path.exists(os.path.join(und, "sparse", "cameras.bin")):
            sys.exit(f"[pose_prior] image_undistorter failed — see {os.path.join(a.out, '04_undistort.log')}")
        k = fitted_distortion(final)
        if k:
            print(f"[pose_prior] fitted distortion k1={k[0]:.4f} k2={k[1]:.4f} p1={k[2]:.5f} p2={k[3]:.5f}; "
                  f"undistorted to PINHOLE")
        dist = os.path.join(a.out, "sparse", "0_distorted")
        shutil.rmtree(dist, ignore_errors=True)
        os.rename(final, dist)
        os.makedirs(final)
        for f in os.listdir(os.path.join(und, "sparse")):
            os.rename(os.path.join(und, "sparse", f), os.path.join(final, f))
        if os.path.islink(link):
            os.remove(link)
        elif os.path.exists(link):
            shutil.rmtree(link)
        os.rename(os.path.join(und, "images"), link)
    if not os.path.exists(link):
        os.symlink(os.path.abspath(a.images), link)
    print(f"[pose_prior] -> {a.out}")


def fitted_distortion(model_dir):
    """k1,k2,p1,p2 of the first OPENCV camera in a binary model, or None."""
    try:
        with open(os.path.join(model_dir, "cameras.bin"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                _cid, model_id, _w, _h = struct.unpack("<iiQQ", f.read(24))
                nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id)
                if nparams is None:
                    return None
                p = struct.unpack(f"<{nparams}d", f.read(8 * nparams))
                if model_id == 4:          # OPENCV: fx fy cx cy k1 k2 p1 p2
                    return p[4:8]
    except OSError:
        return None
    return None


if __name__ == "__main__":
    main()
