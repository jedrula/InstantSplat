"""
filter_sfm_outliers.py — remove cameras whose position is >3σ from the centroid.

Called from video_to_splat.sh with env vars:
  SPARSE_PATH      path to COLMAP binary reconstruction (sparse/0/)
  IMAGE_DIR_PATH   path to the images folder used for training

A single misregistered camera placed thousands of units away from the rest
will inject wrong gradients into gsplat training and scatter all Gaussians.
We detect such outliers and:
  1. Move their image files to <IMAGE_DIR_PATH>/../images_excluded/
  2. Deregister them from the COLMAP reconstruction and write it back (binary)
so the gsplat Parser sees a clean, consistent dataset.
"""

import os, sys, shutil, pycolmap
import numpy as np
from pathlib import Path

sparse = os.environ.get('SPARSE_PATH')
image_dir = os.environ.get('IMAGE_DIR_PATH')

if not sparse or not image_dir:
    print("    filter_sfm_outliers: SPARSE_PATH or IMAGE_DIR_PATH not set, skipping")
    sys.exit(0)

sparse = Path(sparse)
image_dir = Path(image_dir)

r = pycolmap.Reconstruction(str(sparse))
imgs = list(r.images.values())

if len(imgs) < 4:
    print(f"    ✓  {len(imgs)} images — too few to run outlier filter")
    sys.exit(0)

positions = np.array([img.cam_from_world().translation for img in imgs])
center = np.median(positions, axis=0)
dists = np.linalg.norm(positions - center, axis=1)
# Use median + 3*std (robust to the outliers themselves skewing the mean)
threshold = np.median(dists) + 3.0 * dists.std()
outlier_names = {imgs[i].name for i, d in enumerate(dists) if d > threshold}

if not outlier_names:
    print(f"    ✓  All {len(imgs)} cameras within 3σ of centroid (threshold={threshold:.2f})")
    sys.exit(0)

print(f"    ⚠  {len(outlier_names)} outlier camera(s) removed (threshold={threshold:.2f}):")
for i, img in enumerate(imgs):
    if img.name in outlier_names:
        print(f"         {img.name}  (dist={dists[i]:.1f})")

# ── Move image files to excluded dir ─────────────────────────────────────────
excluded_dir = image_dir.parent / "images_excluded"
excluded_dir.mkdir(exist_ok=True)
for name in outlier_names:
    src = image_dir / name
    dst = excluded_dir / name
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"         moved {name} → images_excluded/")

# ── Rebuild reconstruction without outlier images ────────────────────────────
# deregister_frame drops the pose and every observation (and points left with too
# short a track); write_binary then omits the unregistered images entirely.
for img in imgs:
    if img.name in outlier_names:
        r.deregister_frame(img.frame_id)
r.write_binary(str(sparse))

n_kept = len(imgs) - len(outlier_names)
print(f"    ✓  Reconstruction saved with {n_kept} cameras")
