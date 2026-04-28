import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.graph import make_pairs_fps
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_test, load_images, compute_co_vis_masks)
from utils.camera_utils import generate_interpolated_path


# Auto-switch to SparseGA above this frame count to avoid GPU OOM.
# PCO stores ALL pair inference outputs in memory simultaneously: O(N² × H × W).
# At image_size=512: safe up to ~12 frames. At image_size=256: safe up to ~20 frames.
# Above threshold, SparseGA caches pairs to disk (O(1) GPU memory during inference).
# Note: SparseGA reshape bug was fixed — results should now be usable above threshold.
SPARSE_GA_THRESHOLD = 20


@torch.no_grad()
def _compute_sim_matrix(model, images, device):
    """Global cosine-similarity matrix via mean-pooled encoder features."""
    descs = []
    for img_data in images:
        feat = model._encode_image(img_data['img'].to(device),
                                   true_shape=img_data['true_shape'])[0]  # (1, P, D)
        desc = torch.nn.functional.normalize(feat.mean(dim=1), dim=-1)   # (1, D)
        descs.append(desc.cpu())
    descs = torch.cat(descs, dim=0)  # (N, D)
    return (descs @ descs.T).numpy()


def main(source_path, model_path, ckpt_path, device, batch_size, image_size, schedule, lr, niter,
         min_conf_thr, llffhold, n_views, co_vis_dsp, depth_thre, conf_aware_ranking=False,
         focal_avg=False, infer_video=False, max_init_points=None, sparse_pairs=False,
         use_sparse_ga=False, force_dense_ga=False):

    # ---------------- (1) Load model and images ----------------
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    model = AsymmetricMASt3R.from_pretrained(ckpt_path).to(device)
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    if infer_video:
        train_img_files = image_files
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)

    image_files = train_img_files
    images, org_imgs_shape = load_images(image_files, size=image_size)

    # Decide which aligner to use
    do_sparse_ga = (use_sparse_ga or len(images) > SPARSE_GA_THRESHOLD) and not force_dense_ga
    print(f'>> Aligner: {"SparseGA" if do_sparse_ga else "PointCloudOptimizer"} ({len(images)} frames, threshold={SPARSE_GA_THRESHOLD})')

    # Auto-cap init points for SparseGA to avoid 3DGS training OOM
    if do_sparse_ga and max_init_points is None:
        max_init_points = 1_500_000
        print(f'>> Auto-capping max_init_points to {max_init_points:,} (SparseGA mode)')

    start_time = time()
    print(f'>> Making pairs...')
    if sparse_pairs:
        sim_mat = _compute_sim_matrix(model, images, device)
        Na = max(8, len(images) // 2)
        fps_pairs, _ = make_pairs_fps(sim_mat, Na=Na, tokK=4)
        pairs = [(images[i], images[j]) for i, j in fps_pairs]
        pairs += [(images[j], images[i]) for i, j in fps_pairs]  # symmetrize
        print(f'>> Sparse pairs: {len(fps_pairs)} unique ({len(images)} images, Na={Na})')
    else:
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        print(f'>> Complete graph: {len(pairs)//2} unique pairs ({len(images)} images)')

    # ---------------- (2a) SparseGA path (memory-bounded, scales to 50+ frames) ----------------
    if do_sparse_ga:
        print(f'>> SparseGA inference + alignment (pairs cached to disk)...')
        cache_path = str(Path(model_path) / 'sparse_ga_cache')
        image_paths = [str(f) for f in image_files]

        scene = sparse_global_alignment(
            image_paths, pairs, cache_path, model,
            lr1=0.07, niter1=500, lr2=0.014, niter2=200,
            device=device, dtype=torch.float32,
            shared_intrinsics=focal_avg,
        )
        del model
        torch.cuda.empty_cache()

        extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
        focals = to_numpy(scene.get_focals())
        imgs = np.array(scene.imgs)

        # Build intrinsic matrices from focal + principal point
        pps = to_numpy(scene.get_principal_points())  # (N, 2)
        H, W = imgs.shape[1], imgs.shape[2]
        intrinsics = np.array([
            [[f, 0, pps[i, 0]], [0, f, pps[i, 1]], [0, 0, 1]]
            for i, f in enumerate(focals)
        ])

        print(f'>> Densifying point cloud from cached canonical views...')
        pts3d_list, dm_list, conf_list = scene.get_dense_pts3d(clean_depth=True)
        H, W = imgs.shape[1], imgs.shape[2]
        pts3d     = np.array([to_numpy(p).reshape(H, W, 3) for p in pts3d_list])
        depthmaps = np.array([to_numpy(d).reshape(H, W)   for d in dm_list])
        confs     = np.array([to_numpy(c).reshape(H, W)   for c in conf_list])

    # ---------------- (2b) PointCloudOptimizer path (best quality, ≤12 frames) ----------------
    else:
        print(f'>> Inference...')
        output = inference(pairs, model, device, batch_size=1, verbose=True)
        del model
        torch.cuda.empty_cache()
        print(f'>> Global alignment...')
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=300, schedule=schedule, lr=lr, focal_avg=focal_avg)

        extrinsics_w2c = inv(to_numpy(scene.get_im_poses()))
        intrinsics = to_numpy(scene.get_intrinsics())
        focals = to_numpy(scene.get_focals())
        imgs = np.array(scene.imgs)
        pts3d = to_numpy(scene.get_pts3d())
        pts3d = np.array(pts3d)
        depthmaps = to_numpy(scene.im_depthmaps.detach().cpu().numpy())
        values = [param.detach().cpu().numpy() for param in scene.im_conf]
        confs = np.array(values)

    # ---------------- (3) Confidence-aware ranking ----------------
    if conf_aware_ranking:
        print(f'>> Confidence-aware ranking...')
        avg_conf_scores = confs.mean(axis=(1, 2))
        sorted_conf_indices = np.argsort(avg_conf_scores)[::-1]
        print("Sorted indices:", sorted_conf_indices)
        print("Sorted avg confidence:", avg_conf_scores[sorted_conf_indices])
    else:
        sorted_conf_indices = np.arange(n_views)
        print("Sorted indices:", sorted_conf_indices)

    # ---------------- (4) Co-visibility mask ----------------
    # SparseGA returns sparse anchor-based depthmaps incompatible with compute_co_vis_masks;
    # skip masking in that path (it's an optimisation, not required for correctness).
    print(f'>> Calculate the co-visibility mask...')
    if depth_thre > 0 and not do_sparse_ga:
        overlapping_masks = compute_co_vis_masks(sorted_conf_indices, depthmaps, pts3d, intrinsics, extrinsics_w2c, imgs.shape, depth_threshold=depth_thre)
        overlapping_masks = ~overlapping_masks
    else:
        co_vis_dsp = False
        overlapping_masks = None

    end_time = time()
    Train_Time = end_time - start_time
    print(f"Time taken for {n_views} views: {Train_Time} seconds")
    save_time(model_path, '[1] coarse_init_TrainTime', Train_Time)

    # ---------------- (5) Interpolate test poses (novel-view mode only) ----------------
    if not infer_video:
        n_train = len(train_img_files)
        n_test = len(test_img_files)

        if n_train < n_test:
            n_interp = (n_test // (n_train-1)) + 1
            all_inter_pose = []
            for i in range(n_train-1):
                tmp_inter_pose = generate_interpolated_path(poses=extrinsics_w2c[i:i+2], n_interp=n_interp)
                all_inter_pose.append(tmp_inter_pose)
            all_inter_pose = np.concatenate(all_inter_pose, axis=0)
            all_inter_pose = np.concatenate([all_inter_pose, extrinsics_w2c[-1][:3, :].reshape(1, 3, 4)], axis=0)
            indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
            sampled_poses = all_inter_pose[indices]
            sampled_poses = np.array(sampled_poses).reshape(-1, 3, 4)
            assert sampled_poses.shape[0] == n_test
            inter_pose_list = []
            for p in sampled_poses:
                tmp_view = np.eye(4)
                tmp_view[:3, :3] = p[:3, :3]
                tmp_view[:3, 3] = p[:3, 3]
                inter_pose_list.append(tmp_view)
            pose_test_init = np.stack(inter_pose_list, 0)
        else:
            indices = np.linspace(0, extrinsics_w2c.shape[0] - 1, n_test, dtype=int)
            pose_test_init = extrinsics_w2c[indices]

        save_extrinsic(sparse_1_path, pose_test_init, test_img_files, image_suffix)
        test_focals = np.repeat(focals[0], n_test)
        save_intrinsics(sparse_1_path, test_focals, org_imgs_shape, imgs.shape, save_focals=False)

    # ---------------- (6) Save results ----------------
    focals = np.repeat(focals[0], n_views)
    print(f'>> Saving results...')
    end_time = time()
    save_time(model_path, '[1] init_geo', end_time - start_time)
    save_extrinsic(sparse_0_path, extrinsics_w2c, image_files, image_suffix)
    save_intrinsics(sparse_0_path, focals, org_imgs_shape, imgs.shape, save_focals=True)
    save_pts_kwargs = dict(use_masks=co_vis_dsp, save_all_pts=True, save_txt_path=model_path, depth_threshold=depth_thre)
    if max_init_points is not None:
        save_pts_kwargs['max_pts_num'] = max_init_points
    pts_num = save_points3D(sparse_0_path, imgs, pts3d, confs.reshape(pts3d.shape[0], -1), overlapping_masks, **save_pts_kwargs)
    if overlapping_masks is not None:
        save_images_and_masks(sparse_0_path, n_views, imgs, overlapping_masks, image_files, image_suffix)
    print(f'[INFO] MASt3R Reconstruction is successfully converted to COLMAP files in: {str(sparse_0_path)}')
    print(f'[INFO] Number of points: {pts3d.reshape(-1, 3).shape[0]}')
    print(f'[INFO] Number of points after downsampling: {pts_num}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--source_path', '-s', type=str, required=True)
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str,
        default='./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--schedule', type=str, default='cosine')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--min_conf_thr', type=float, default=5)
    parser.add_argument('--llffhold', type=int, default=8)
    parser.add_argument('--n_views', type=int, default=3)
    parser.add_argument('--focal_avg', action='store_true')
    parser.add_argument('--conf_aware_ranking', action='store_true')
    parser.add_argument('--co_vis_dsp', action='store_true')
    parser.add_argument('--depth_thre', type=float, default=0.01)
    parser.add_argument('--infer_video', action='store_true')
    parser.add_argument('--max_init_points', type=int, default=None,
                        help='Cap initial point cloud size (confidence-weighted downsample)')
    parser.add_argument('--sparse_pairs', action='store_true',
                        help='Use sparse FPS retrieval pairing instead of complete graph')
    parser.add_argument('--sparse_ga', action='store_true',
                        help=f'Force SparseGA aligner (auto-enabled above {SPARSE_GA_THRESHOLD} frames)')
    parser.add_argument('--no_sparse_ga', action='store_true',
                        help='Force PointCloudOptimizer even above threshold (may OOM)')

    args = parser.parse_args()
    main(args.source_path, args.model_path, args.ckpt_path, args.device, args.batch_size, args.image_size,
         args.schedule, args.lr, args.niter, args.min_conf_thr, args.llffhold, args.n_views,
         args.co_vis_dsp, args.depth_thre, args.conf_aware_ranking, args.focal_avg, args.infer_video,
         args.max_init_points, args.sparse_pairs,
         use_sparse_ga=args.sparse_ga, force_dense_ga=args.no_sparse_ga)
