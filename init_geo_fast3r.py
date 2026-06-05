import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time
from PIL import Image

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.dust3r.utils.image import load_images as fast3r_load_images
from fast3r.dust3r.inference_multiview import inference as fast3r_inference
from utils.sfm_utils import (
    save_intrinsics, save_extrinsic, save_points3D, save_time,
    init_filestructure, get_sorted_image_files, compute_co_vis_masks,
)

FAST3R_CKPT = "jedyang97/Fast3R_ViT_Large_512"


def refine_poses_with_colmap_ba(image_files, image_dir, poses_c2w, focal_inf, H_inf, W_inf,
                                 work_dir, n_ba_iters=50):
    """
    Refine Fast3R's initial camera poses using COLMAP feature matching + pycolmap BA.

    Fast3R's independent per-view PnP produces globally inconsistent poses for
    large-motion scenes. This function:
      1. Extracts SIFT features via pycolmap (CPU, no Qt/OpenGL needed)
      2. Runs exhaustive matching
      3. Seeds a pycolmap Reconstruction with Fast3R's initial poses
      4. Triangulates 3D points from feature matches + initial poses
      5. Runs bundle adjustment to get globally consistent poses

    Returns a list of N refined c2w 4×4 numpy matrices (same order as image_files),
    or the original poses_c2w if BA fails or triangulates too few points.
    """
    try:
        import pycolmap
        from scipy.spatial.transform import Rotation as ScipyRotation
    except ImportError as e:
        print(f'[BA] pycolmap/scipy not available ({e}), skipping BA')
        return poses_c2w

    N = len(image_files)
    image_names = [Path(f).name for f in image_files]
    ba_db = Path(work_dir) / 'colmap_ba.db'
    ba_db.unlink(missing_ok=True)

    # ── 1. Feature extraction (CPU, no OpenGL) ────────────────────────────────
    print(f'[BA] Extracting SIFT features (CPU) for {N} images...')
    t0 = time()
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = 'PINHOLE'
    reader_opts.camera_params = f'{focal_inf},{focal_inf},{W_inf/2},{H_inf/2}'

    extr_opts = pycolmap.FeatureExtractionOptions()
    extr_opts.use_gpu = False

    pycolmap.extract_features(
        ba_db, image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extr_opts,
        device=pycolmap.Device.cpu,
    )
    print(f'[BA]   extraction: {time()-t0:.1f}s')

    # ── 2. Exhaustive feature matching (CPU by default) ───────────────────────
    print('[BA] Exhaustive feature matching...')
    t0 = time()
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.use_gpu = False
    if N > 50:
        pycolmap.match_sequential(ba_db, matching_options=match_opts)
    else:
        pycolmap.match_exhaustive(ba_db, matching_options=match_opts)
    print(f'[BA]   matching: {time()-t0:.1f}s')

    # ── 3. Seed reconstruction with Fast3R poses ──────────────────────────────
    # COLMAP databases are SQLite — pycolmap.Database is pure virtual in 4.x
    import sqlite3
    conn = sqlite3.connect(str(ba_db))
    rows = conn.execute("SELECT image_id, name FROM images").fetchall()
    cam_rows = conn.execute("SELECT camera_id, model, width, height, params FROM cameras").fetchall()
    conn.close()
    db_id_by_name = {name: img_id for img_id, name in rows}

    recon = pycolmap.Reconstruction()
    # Reconstruct the shared PINHOLE camera from the database
    cam_id_db, cam_model, cam_w, cam_h, cam_params_blob = cam_rows[0]
    # params_blob is binary: 4 doubles (fx, fy, cx, cy) for PINHOLE
    import struct
    cam_params = list(struct.unpack(f'{len(cam_params_blob)//8}d', cam_params_blob))
    db_cam = pycolmap.Camera()
    db_cam.camera_id = cam_id_db   # must set ID before add_camera (returns None)
    db_cam.model = pycolmap.CameraModelId.PINHOLE
    db_cam.width = cam_w
    db_cam.height = cam_h
    db_cam.params = cam_params
    # add_camera_with_trivial_rig creates camera + rig (rig_id=camera_id, identity sensor-to-rig)
    # required by add_image_with_trivial_frame which looks up rig by camera_id
    recon.add_camera_with_trivial_rig(db_cam)
    cam_id = cam_id_db

    # name → Fast3R c2w pose
    name_to_c2w = {Path(f).name: poses_c2w[i] for i, f in enumerate(image_files)}
    skipped = []
    for name, img_id in db_id_by_name.items():
        if name not in name_to_c2w:
            skipped.append(name)
            continue
        c2w = np.array(name_to_c2w[name])
        w2c = np.linalg.inv(c2w)
        R_mat = w2c[:3, :3]
        t_vec = w2c[:3, 3]
        q_xyzw = ScipyRotation.from_matrix(R_mat).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(q_wxyz), t_vec)
        img_obj = pycolmap.Image(name=name, camera_id=cam_id, image_id=img_id)
        # add_image_with_trivial_frame overload 2: creates frame+rig and sets pose
        recon.add_image_with_trivial_frame(img_obj, cam_from_world)

    if skipped:
        print(f'[BA]   WARNING: {len(skipped)} image(s) not in Fast3R output, skipped')

    # ── 4. Triangulate 3D points given initial poses ──────────────────────────
    print(f'[BA] Triangulating points ({len(recon.reg_image_ids())} registered images)...')
    t0 = time()
    ba_sparse = Path(work_dir) / 'colmap_ba_sparse'
    ba_sparse.mkdir(exist_ok=True)
    recon = pycolmap.triangulate_points(recon, ba_db, image_dir, ba_sparse)
    n_pts = len(recon.points3D)
    print(f'[BA]   triangulated {n_pts} points in {time()-t0:.1f}s')

    if n_pts < 50:
        print(f'[BA]   too few points ({n_pts}), keeping Fast3R poses')
        return poses_c2w

    # ── 5. Bundle adjustment ──────────────────────────────────────────────────
    print(f'[BA] Bundle adjustment ({n_ba_iters} max iters)...')
    t0 = time()
    ba_opts = pycolmap.BundleAdjustmentOptions()
    ba_opts.ceres.solver_options.max_num_iterations = n_ba_iters
    pycolmap.bundle_adjustment(recon, ba_opts)
    reproj_err = recon.compute_mean_reprojection_error()
    print(f'[BA]   done in {time()-t0:.1f}s, mean reproj error: {reproj_err:.3f}px')

    # ── 6. Extract refined poses in original image order ─────────────────────
    id_to_img = {img.image_id: img for img in recon.images.values()}

    refined_poses = []
    n_refined = 0
    for i, f in enumerate(image_files):
        name = Path(f).name
        img_id = db_id_by_name.get(name)
        if img_id is None or img_id not in id_to_img:
            refined_poses.append(poses_c2w[i])  # fallback
            continue
        img_obj = id_to_img[img_id]
        # cam_from_world() is a method in pycolmap 4.x returning a w2c Rigid3d
        cfw = img_obj.cam_from_world()
        w2c_rot = cfw.rotation.matrix()
        w2c_t = cfw.translation
        w2c = np.eye(4)
        w2c[:3, :3] = w2c_rot
        w2c[:3, 3] = w2c_t
        refined_poses.append(np.linalg.inv(w2c))
        n_refined += 1

    print(f'[BA]   refined {n_refined}/{N} poses')
    ba_db.unlink(missing_ok=True)
    return refined_poses


def main(source_path, model_path, device, image_size, n_views,
         co_vis_dsp, depth_thre, max_init_points, colmap_ba):

    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)

    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)

    # Map image_size to Fast3R inference size. Any value != 224 takes the "resize long side"
    # path in Fast3R's loader, which preserves aspect ratio for any input. 224 is special-cased
    # to square-crop — never use it as it destroys the camera model for non-square images.
    fast3r_size = 512 if image_size >= 384 else 256

    # original image dimensions for save_intrinsics scale factor
    img0 = Image.open(image_files[0])
    org_imgs_shape = img0.size  # (W, H)
    img0.close()

    device = torch.device(device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f'>> Fast3R: size={fast3r_size}, dtype={dtype}, {len(image_files)} images')

    model = Fast3R.from_pretrained(FAST3R_CKPT).to(device)
    model.eval()

    start_time = time()

    images = fast3r_load_images(image_files, size=fast3r_size, verbose=True)

    with torch.no_grad():
        result = fast3r_inference(images, model, device, dtype=dtype, verbose=True, profiling=False)
    output_dict = result[0] if isinstance(result, tuple) else result

    preds = output_dict['preds']
    views_out = output_dict['views']

    poses_c2w_all, estimated_focals_all = MultiViewDUSt3RLitModule.estimate_camera_poses(
        preds, niter_PnP=10,
        focal_length_estimation_method='first_view_from_global_head',
    )
    poses_c2w = poses_c2w_all[0]   # list of N 4x4 numpy c2w matrices
    focals     = estimated_focals_all[0]  # list of N floats

    del model
    torch.cuda.empty_cache()

    N = len(image_files)
    H = preds[0]['pts3d_in_other_view'][0].shape[0]
    W = preds[0]['pts3d_in_other_view'][0].shape[1]

    # Fast3R's depth-based focal estimator often overestimates for scenes with
    # little depth variation (flat walls, outdoor). Cap at 1.5× the 60°-FOV prior
    # by scaling pts3d x,y and re-running PnP with the corrected focal.
    focal_60deg = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))
    focal_estimated = float(focals[0])
    if focal_estimated > focal_60deg * 1.5:
        focal_target = focal_60deg
        xy_scale = focal_estimated / focal_target  # >1: expand lateral spread
        print(f'[INFO] Fast3R focal {focal_estimated:.1f}px > 1.5× 60°-prior ({focal_60deg:.1f}px). '
              f'Rescaling pts3d x,y by {xy_scale:.3f} and re-estimating poses...')
        for pred in preds:
            pred['pts3d_in_other_view'][..., :2] *= xy_scale
        poses_c2w_all, estimated_focals_all = MultiViewDUSt3RLitModule.estimate_camera_poses(
            preds, niter_PnP=10,
            focal_length_estimation_method='first_view_from_global_head',
        )
        poses_c2w = poses_c2w_all[0]
        focals     = estimated_focals_all[0]
        print(f'[INFO] Corrected focal: {float(focals[0]):.1f}px (inference)')

    # ── Optional: COLMAP BA to get globally consistent poses ──────────────────
    # Fast3R's independent per-view PnP accumulates drift for large-motion scenes.
    # COLMAP BA uses real 2D-2D SIFT correspondences to jointly optimize all poses,
    # bypassing Fast3R's view-0-reference bias.
    if colmap_ba:
        focal_inf = float(focals[0])
        poses_c2w = refine_poses_with_colmap_ba(
            image_files, image_dir, poses_c2w,
            focal_inf=focal_inf, H_inf=H, W_inf=W,
            work_dir=source_path,
        )

    pts3d_arr = np.stack([preds[i]['pts3d_in_other_view'][0].cpu().float().numpy() for i in range(N)])  # (N,H,W,3)
    confs_arr = np.stack([preds[i]['conf'][0].cpu().float().numpy()                for i in range(N)])  # (N,H,W)
    imgs_arr  = np.stack([
        ((views_out[i]['img'][0].permute(1, 2, 0).cpu().float().numpy() + 1) / 2).clip(0, 1)
        for i in range(N)
    ])  # (N,H,W,3)

    # c2w → w2c
    extrinsics_w2c = np.stack([np.linalg.inv(np.array(c2w)) for c2w in poses_c2w])  # (N,4,4)
    focals_arr = np.repeat(float(focals[0]), N)

    # co-visibility mask (use Z channel of pts3d as depth proxy)
    if depth_thre > 0 and co_vis_dsp:
        intrinsics = np.array([
            [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]] for f in focals_arr
        ])
        depthmaps = pts3d_arr[..., 2]
        overlapping_masks = compute_co_vis_masks(
            np.arange(N), depthmaps, pts3d_arr, intrinsics, extrinsics_w2c,
            imgs_arr.shape, depth_threshold=depth_thre,
        )
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None

    elapsed = time() - start_time
    print(f"Time taken for {n_views} views: {elapsed:.1f} seconds")
    save_time(model_path, '[1] coarse_init_TrainTime', elapsed)

    print('>> Saving results...')
    save_time(model_path, '[1] init_geo', time() - start_time)
    save_extrinsic(sparse_0_path, extrinsics_w2c, image_files, image_suffix)
    save_intrinsics(sparse_0_path, focals_arr, org_imgs_shape, imgs_arr.shape, save_focals=True)

    save_pts_kwargs = dict(
        use_masks=co_vis_dsp, save_all_pts=True,
        save_txt_path=model_path, depth_threshold=depth_thre,
    )
    if max_init_points is not None:
        save_pts_kwargs['max_pts_num'] = max_init_points

    pts_num = save_points3D(
        sparse_0_path, imgs_arr, pts3d_arr,
        confs_arr.reshape(N, -1), overlapping_masks,
        **save_pts_kwargs,
    )
    print(f'[INFO] Fast3R Reconstruction converted to COLMAP files in: {sparse_0_path}')
    print(f'[INFO] Number of points: {pts3d_arr.reshape(-1, 3).shape[0]}')
    print(f'[INFO] Number of points after downsampling: {pts_num}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path', required=True)
    parser.add_argument('-m', '--model_path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_views', type=int, default=3)
    parser.add_argument('--co_vis_dsp', action='store_true')
    parser.add_argument('--depth_thre', type=float, default=0.01)
    parser.add_argument('--max_init_points', type=int, default=None)
    parser.add_argument('--colmap_ba', action='store_true',
                        help='Refine Fast3R poses with COLMAP feature matching + bundle adjustment')
    parser.add_argument('--no_point_cap', action='store_true',
                        help='No-op for Fast3R (point cap only applies to MASt3R SparseGA path)')
    args = parser.parse_args()
    main(args.source_path, args.model_path, args.device, args.image_size,
         args.n_views, args.co_vis_dsp, args.depth_thre, args.max_init_points,
         args.colmap_ba)
