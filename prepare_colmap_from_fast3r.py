"""
Helper script: load pre-saved Fast3R numpy arrays and write COLMAP sparse files.
Run with the InstantSplat conda python.

Usage:
  python prepare_colmap_from_fast3r.py <work_dir>

Expects these files in <work_dir>/fast3r_arrays/:
  pts3d_arr.npy      (N, H, W, 3)
  confs_arr.npy      (N, H, W)
  imgs_arr.npy       (N, H, W, 3)  float32 in [0,1]
  extrinsics_w2c.npy (N, 4, 4)
  focals_arr.npy     (N,)
  meta.json          {"n_views": N, "org_w": W, "org_h": H}

Outputs COLMAP sparse files into <work_dir>/splat_source/sparse_{n_views}/0/
"""
import sys
import json
import numpy as np
from pathlib import Path

work_dir = Path(sys.argv[1])
arr_dir = work_dir / "fast3r_arrays"

# Load arrays
pts3d_arr      = np.load(arr_dir / "pts3d_arr.npy")
confs_arr      = np.load(arr_dir / "confs_arr.npy")
imgs_arr       = np.load(arr_dir / "imgs_arr.npy")
extrinsics_w2c = np.load(arr_dir / "extrinsics_w2c.npy")
focals_arr     = np.load(arr_dir / "focals_arr.npy")
with open(arr_dir / "meta.json") as f:
    meta = json.load(f)

N = meta["n_views"]
org_w = meta["org_w"]
org_h = meta["org_h"]
org_imgs_shape = (org_w, org_h)  # PIL .size is (width, height)

source_path = work_dir / "splat_source"
model_path  = work_dir / "splat_model"

# Import InstantSplat utils (simple_knn is available in this env)
sys.path.insert(0, str(Path(__file__).parent))
from utils.sfm_utils import (
    save_intrinsics, save_extrinsic, save_points3D,
    init_filestructure, get_sorted_image_files,
)

_, sparse_0_path, _ = init_filestructure(source_path, N)
img_out_dir = source_path / "images"
image_files_sorted, image_suffix = get_sorted_image_files(img_out_dir)

save_extrinsic(sparse_0_path, extrinsics_w2c, image_files_sorted, image_suffix)
save_intrinsics(sparse_0_path, focals_arr, org_imgs_shape, imgs_arr.shape, save_focals=True)
save_points3D(
    sparse_0_path, imgs_arr, pts3d_arr,
    confs_arr.reshape(N, -1), None,
    use_masks=False, save_all_pts=True,
    save_txt_path=str(model_path),
)

print(f"[prepare_colmap] Done. sparse files → {sparse_0_path}")
