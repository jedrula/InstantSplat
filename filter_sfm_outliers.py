"""
filter_sfm_outliers.py — remove cameras whose position is >3σ from the centroid.

Called from video_to_splat_glomap.sh with env vars:
  SPARSE_PATH      path to COLMAP binary reconstruction (sparse/0/)
  IMAGE_DIR_PATH   path to the images folder used for training

A single misregistered camera placed thousands of units away from the rest
will inject wrong gradients into gsplat training and scatter all Gaussians.
We detect such outliers and:
  1. Move their image files to <IMAGE_DIR_PATH>/../images_excluded/
  2. Remove them from the COLMAP reconstruction (write_text + filter + write_binary)
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
# Export to text, filter images.txt (2 lines per image), filter points3D.txt
# (remove observations from excluded image_ids), import back to binary.
txt_path = sparse.parent / "0_txt"
txt_path.mkdir(exist_ok=True)
r.write_text(str(txt_path))

outlier_ids = {img.image_id for img in imgs if img.name in outlier_names}

# Filter images.txt (two lines per image: header line + points2D line)
images_txt = txt_path / "images.txt"
lines = images_txt.read_text().splitlines(keepends=True)
filtered = []
skip_next = False
for line in lines:
    if line.startswith('#'):
        filtered.append(line)
        continue
    if skip_next:
        skip_next = False
        continue
    # Header line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    parts = line.split()
    if parts and parts[0].isdigit() and int(parts[0]) in outlier_ids:
        skip_next = True  # also skip the following points2D line
        continue
    filtered.append(line)
images_txt.write_text(''.join(filtered))

# Filter points3D.txt: remove observations referencing outlier image_ids
points_txt = txt_path / "points3D.txt"
p3d_lines = points_txt.read_text().splitlines(keepends=True)
filtered_pts = []
for line in p3d_lines:
    if line.startswith('#') or not line.strip():
        filtered_pts.append(line)
        continue
    # POINT3D_ID X Y Z R G B ERROR [TRACK: IMAGE_ID POINT2D_IDX ...]
    parts = line.split()
    fixed = parts[:8]  # ID X Y Z R G B ERROR
    track = parts[8:]  # pairs: image_id point2d_idx ...
    new_track = []
    for j in range(0, len(track) - 1, 2):
        if int(track[j]) not in outlier_ids:
            new_track.extend([track[j], track[j+1]])
    if new_track:  # keep point only if it still has observers
        filtered_pts.append(' '.join(fixed + new_track) + '\n')
points_txt.write_text(''.join(filtered_pts))

# Reload and write back to binary
r2 = pycolmap.Reconstruction()
r2.read_text(str(txt_path))
r2.write_binary(str(sparse))
shutil.rmtree(str(txt_path))

n_kept = len(imgs) - len(outlier_names)
print(f"    ✓  Reconstruction saved with {n_kept} cameras")
