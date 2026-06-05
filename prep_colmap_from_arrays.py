"""
Helper script — run with the InstantSplat conda python.

Reads numpy arrays saved by Fast3R's demo.py and writes COLMAP-format
sparse files so train.py can consume them.

Usage:
    python prep_colmap_from_arrays.py <work_dir>

Expected files in <work_dir>:
    pts3d.npy       (N, H, W, 3)
    confs.npy       (N, H, W)
    imgs.npy        (N, H, W, 3)  float32 in [0,1]
    extrinsics.npy  (N, 4, 4)  w2c
    focals.npy      (N,)
    org_W.txt       original image width  (PIL .size[0])
    org_H.txt       original image height (PIL .size[1])
    n_views.txt     N as int
    img_dir/        directory of images (already copied)

Outputs written to <work_dir>/sparse_{n_views}/0/
"""
import sys
import numpy as np
from pathlib import Path

work_dir = Path(sys.argv[1])
source_path = work_dir / "splat_source"
model_path  = work_dir / "splat_model"
img_dir     = source_path / "images"

pts3d_arr      = np.load(work_dir / "pts3d.npy")
confs_arr      = np.load(work_dir / "confs.npy")
imgs_arr       = np.load(work_dir / "imgs.npy")
extrinsics_w2c = np.load(work_dir / "extrinsics.npy")
focals_arr     = np.load(work_dir / "focals.npy")
org_W = int((work_dir / "org_W.txt").read_text().strip())
org_H = int((work_dir / "org_H.txt").read_text().strip())
n_views = int((work_dir / "n_views.txt").read_text().strip())

sys.path.insert(0, str(Path(__file__).parent))

from utils.sfm_utils import (
    save_intrinsics, save_extrinsic, save_points3D,
    init_filestructure, get_sorted_image_files,
)

_, sparse_0_path, _ = init_filestructure(source_path, n_views)
image_files_sorted, image_suffix = get_sorted_image_files(img_dir)
org_imgs_shape = (org_W, org_H)  # PIL .size is (width, height)

N = len(pts3d_arr)

save_extrinsic(sparse_0_path, extrinsics_w2c, image_files_sorted, image_suffix)
save_intrinsics(sparse_0_path, focals_arr, org_imgs_shape, imgs_arr.shape, save_focals=True)
save_points3D(
    sparse_0_path, imgs_arr, pts3d_arr,
    confs_arr.reshape(N, -1), None,
    use_masks=False, save_all_pts=True,
    save_txt_path=str(model_path),
)

print(f"[prep_colmap] Done. sparse files → {sparse_0_path}")
