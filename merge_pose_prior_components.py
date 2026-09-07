#!/usr/bin/env python3
"""Merge every component of a pose_prior_mapper reconstruction into one model.

WHY THIS EXISTS. `pose_prior_mapper` routinely emits several disconnected components, and
`arkit_pose_prior.py` used to keep only the largest — throwing away roughly half the capture.
On b36e3755 that cost 275 of 515 frames (run D, spatial matching: 240 kept, 7 components
discarded holding 219 more frames).

WHY IT IS SOUND HERE AND NOT IN GENERAL. Ordinary SfM components each have their own gauge
(arbitrary rotation, translation and SCALE), so concatenating them is meaningless. Pose-prior
components do not: every one of them was anchored to the same external ARKit metric frame, so
they are already mutually registered even when they share no images. Measured on run D — one
sim3 fitted on component 0 alone, applied unchanged to the other six:

    comp  frames  median  p90     verdict
    0     240     16.5cm  43.1cm  same frame
    1      95     21.7cm  26.5cm  same frame
    2      37     11.6cm  17.7cm  same frame
    3      23     15.6cm  17.3cm  same frame
    4      27      9.7cm  11.4cm  same frame
    5      10     15.7cm  16.5cm  same frame
    6      27     14.7cm  17.2cm  same frame

Scored after merging, against the same ARKit trajectory:

    model        frames        scale   median   p90      max      >1m
    comp 0 only  240/515 (47%) 0.9858  16.5cm   43.1cm   52.6cm   0
    MERGED       439/515 (85%) 1.0006   9.4cm   32.6cm   47.4cm   0

i.e. 83% more frames AND better on every accuracy column. Merged scale 1.0006 — the
reconstruction is metric to 0.06%, because the priors are.

439, not 459: components 3 and 6 register 20 of the same images twice (their camera centres
agree to 0.6cm median, so they really are the same frames). Summing component sizes
double-counts them; see the duplicate handling in merge().

That property is an assumption about the input, so this script CHECKS it rather than trusting
it: --traj enables a per-component residual gate against the ARKit positions and the merge
ABORTS if any component disagrees with the base component's frame. Run without --traj only on
a model you have already verified.

ID SPACES. image_ids and frame_ids come from the database and are therefore global and disjoint
across components — verified, and re-checked at runtime. point3D_ids are per-component and DO
collide, so they are offset. Track elements reference image_ids, which need no remapping.
"""
import argparse, os, re, shutil, sys
from pathlib import Path

import numpy as np
import pycolmap


def umeyama(X, Y):
    """Least-squares similarity transform mapping X onto Y."""
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
    """Umeyama with trimming. A single folded block otherwise drags the whole fit."""
    s, R, t = umeyama(X, Y)
    for _ in range(iters):
        keep = np.linalg.norm(apply_sim3(s, R, t, X) - Y, axis=1) < thresh
        if keep.sum() < 4:
            break
        s, R, t = umeyama(X[keep], Y[keep])
    return s, R, t


def frame_index(name):
    """Frame ordinal from an image filename; the traj is positional, line i <-> frame i."""
    m = re.search(r"(\d{4})\.(?:jpg|jpeg|png)$", name, re.I)
    return int(m.group(1)) if m else None


def centres_by_frame(rec):
    out = {}
    for im in rec.images.values():
        i = frame_index(im.name)
        if i is not None:
            out[i] = im.projection_center()
    return out


def load_components(sparse_dir):
    comps = []
    for d in sorted(Path(sparse_dir).iterdir()):
        if not (d / "images.bin").exists() and not (d / "images.txt").exists():
            continue
        if not d.name.isdigit():
            continue
        comps.append((d, pycolmap.Reconstruction(str(d))))
    return comps


def check_same_frame(comps, base_idx, traj_path, max_median):
    """Gate: every component must land in the base component's frame under ONE sim3."""
    pos = np.array([[float(x) for x in l.split()[4:7]]
                    for l in open(traj_path) if len(l.split()) == 7])
    base_dir, base_rec = comps[base_idx]
    c0 = centres_by_frame(base_rec)
    ids0 = np.array(sorted(i for i in c0 if i < len(pos)))
    if len(ids0) < 8:
        sys.exit(f"[merge] base component has only {len(ids0)} frames matchable to the traj; "
                 f"cannot verify the shared frame. Refusing to merge.")
    s, R, t = robust_sim3(np.array([c0[i] for i in ids0]), pos[ids0])
    print(f"[merge] sim3 fitted on component {base_dir.name} only (scale {s:.4f})")
    print(f"[merge] {'comp':>5} {'frames':>7} {'median cm':>10} {'p90 cm':>9}  verdict")

    bad = []
    for d, rec in comps:
        cen = centres_by_frame(rec)
        ids = np.array(sorted(i for i in cen if i < len(pos)))
        if len(ids) == 0:
            bad.append((d.name, "no frames matchable to traj"))
            continue
        res = np.linalg.norm(apply_sim3(s, R, t, np.array([cen[i] for i in ids])) - pos[ids], axis=1)
        med = float(np.median(res))
        ok = med < max_median
        print(f"[merge] {d.name:>5} {len(ids):7d} {med*100:10.1f} "
              f"{np.percentile(res, 90)*100:9.1f}  "
              f"{'same frame' if ok else 'DIFFERENT FRAME'}")
        if not ok:
            bad.append((d.name, f"median {med*100:.1f} cm > {max_median*100:.0f} cm"))
    if bad:
        for name, why in bad:
            print(f"[merge] component {name}: {why}", file=sys.stderr)
        sys.exit("[merge] components do not share one metric frame — refusing to merge. "
                 "Concatenating models with different gauges would corrupt the reconstruction.")


def merge(comps, base_idx):
    base_dir, base = comps[base_idx]
    out = pycolmap.Reconstruction(str(base_dir))

    seen_images = set(out.images)
    next_pt = (max(out.points3D) + 1) if out.num_points3D() else 1

    # (image_id, point2D_idx) slots already bound to a point3D. A keypoint can reference only
    # one 3-D point, so a later component's track element for an already-claimed slot must be
    # dropped rather than overwriting the earlier one.
    claimed = set()
    for im in out.images.values():
        for idx, p in enumerate(im.points2D):
            if p.has_point3D():
                claimed.add((im.image_id, idx))

    dup_images = dropped_els = dropped_pts = 0

    for i, (d, rec) in enumerate(comps):
        if i == base_idx:
            continue
        clash = seen_images & set(rec.images)
        if clash:
            # pose_prior_mapper CAN register the same image into two components. On b36e3755
            # run D that was components 3 and 6 sharing 20 images, whose camera centres agreed
            # to 0.6 cm median / 1.2 cm max — genuinely the same frame, registered twice. Keep
            # the copy already merged (it came from the larger component) and skip the second.
            dup_images += len(clash)
            print(f"[merge]   component {d.name}: {len(clash)} image(s) already merged "
                  f"(e.g. {sorted(clash)[:4]}) — keeping the first copy")

        for rig in rec.rigs.values():
            if not out.exists_rig(rig.rig_id):
                out.add_rig(rig)
        for cam in rec.cameras.values():
            if not out.exists_camera(cam.camera_id):
                out.add_camera(cam)

        offset = next_pt - 1
        # A Frame/Image read from one Reconstruction holds a raw pointer to THAT
        # reconstruction's Rig/Camera object, and add_frame/add_image assert the pointer
        # matches. reset_rig_ptr() clears it on a Frame; Image exposes no equivalent, so
        # rebuild the Image from its fields instead of re-parenting it.
        for frame in rec.frames.values():
            if not out.exists_frame(frame.frame_id):
                frame.reset_rig_ptr()
                out.add_frame(frame)
        added_imgs = 0
        for im in rec.images.values():
            if im.image_id in seen_images:
                continue
            pts2d = im.points2D
            for idx, p in enumerate(pts2d):
                if p.has_point3D():
                    p.point3D_id = p.point3D_id + offset
                    claimed.add((im.image_id, idx))
            new_im = pycolmap.Image(name=im.name, points2D=pts2d,
                                    camera_id=im.camera_id, image_id=im.image_id)
            new_im.frame_id = im.frame_id
            out.add_image(new_im)
            seen_images.add(im.image_id)
            added_imgs += 1
        for frame in rec.frames.values():
            if frame.has_pose() and not out.frame(frame.frame_id).has_pose():
                out.register_frame(frame.frame_id)

        added_pts = 0
        for pid, pt in rec.points3D.items():
            track = pycolmap.Track()
            for el in pt.track.elements:
                # Skip observations in images we did not take from this component, and slots
                # already bound by an earlier component.
                if el.image_id not in seen_images or (el.image_id, el.point2D_idx) not in claimed:
                    dropped_els += 1
                    continue
                if out.images[el.image_id].points2D[el.point2D_idx].point3D_id != pid + offset:
                    dropped_els += 1
                    continue
                track.add_element(el.image_id, el.point2D_idx)
            if track.length() < 2:
                # A 3-D point needs two views to be a triangulation. Unbind whatever slots it
                # still held so they do not point at a point that will not exist.
                for el in track.elements:
                    out.images[el.image_id].points2D[el.point2D_idx].point3D_id = \
                        pycolmap.INVALID_POINT3D_ID
                dropped_pts += 1
                continue
            new = pycolmap.Point3D()
            new.xyz, new.color, new.error = pt.xyz, pt.color, pt.error
            new.track = track
            out.add_point3D_with_id(pid + offset, new)
            added_pts += 1
        next_pt = offset + (max(rec.points3D) if rec.num_points3D() else 0) + 1
        print(f"[merge]   + component {d.name}: +{added_imgs} images, "
              f"+{added_pts:,} points (pt3D ids +{offset})")

    if dup_images or dropped_els or dropped_pts:
        print(f"[merge] reconciliation: {dup_images} duplicate image(s) skipped, "
              f"{dropped_els:,} track element(s) and {dropped_pts:,} point(s) dropped")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True, help="dir holding numbered components (0/, 1/, ...)")
    ap.add_argument("--out", required=True, help="output model dir")
    ap.add_argument("--traj", default="", help="frames.traj — enables the shared-frame gate. "
                                               "Strongly recommended.")
    ap.add_argument("--max-median-error", type=float, default=0.5,
                    help="metres; a component whose median residual under the base component's "
                         "sim3 exceeds this is treated as a different gauge (default 0.5)")
    a = ap.parse_args()

    comps = load_components(a.sparse)
    if not comps:
        sys.exit(f"[merge] no components under {a.sparse}")
    print(f"[merge] {len(comps)} component(s) in {a.sparse}")
    for d, rec in comps:
        print(f"[merge]   {d.name}: {rec.num_images():4d} images, {rec.num_points3D():>8,} points")
    base_idx = max(range(len(comps)), key=lambda i: comps[i][1].num_images())
    print(f"[merge] base = component {comps[base_idx][0].name} "
          f"({comps[base_idx][1].num_images()} images)")

    if len(comps) == 1:
        print("[merge] single component — nothing to merge")
        merged = comps[0][1]
    else:
        if a.traj:
            check_same_frame(comps, base_idx, a.traj, a.max_median_error)
        else:
            print("[merge] WARNING: no --traj, shared-frame gate SKIPPED", file=sys.stderr)
        merged = merge(comps, base_idx)

    os.makedirs(a.out, exist_ok=True)
    merged.write(a.out)
    merged.write_text(a.out)
    print(f"[merge] -> {a.out}")
    print(f"[merge] {merged.num_images()} images, {merged.num_points3D():,} points, "
          f"mean track {merged.compute_mean_track_length():.2f}, "
          f"mean reproj {merged.compute_mean_reprojection_error():.3f} px")


if __name__ == "__main__":
    main()
