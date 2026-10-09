#!/usr/bin/env python3
"""Delete look-alike pairs from a COLMAP database with the Doppelgangers++ classifier, before the mapper runs.

Why: on repetitive scenes (block2ha's identical facades) retrieval matching verifies pairs between DIFFERENT
buildings -- 6529 of 42936 verified pairs on f8e7c5c5 were >25 m apart -- and GLOMAP folds on them. Probed on
300 of that run's pairs labelled by the sim's true poses, D++ (CVPR 2025) scored AUC 0.985: at 0.8 it kept 99%
of real pairs and removed 86% of look-alikes (experiments/doppelgangers_ab/probe_pairs.py).

Cost is 0.55 s/pair, so not every pair is scored. The capture is a walk in file-name order:
  * pairs within WINDOW photos of each other in the walk are kept unscored (they are the walk's own chain);
  * the rest are grouped into BLOCK x BLOCK blocks of the walk (photos a..a+9 against b..b+9) -- a look-alike
    run is one block -- and the block's strongest pair (most inliers) is scored; a block under THRESHOLD loses
    ALL its pairs, from `matches` and `two_view_geometries`, as D++'s own remove_doppelgangers does.
40x40 (802 photos): 1635 blocks ~15 min; 100x100 (2006): 7702 blocks ~70 min.

LICENSE: D++ is CC BY-NC-SA 4.0 (built on MASt3R) -- research use only, do not ship.

usage: dg_filter.py DATABASE IMAGE_DIR [--threshold 0.8] [--window 10] [--block 10]
"""
import argparse, collections, sqlite3, sys, time
import numpy as np, torch
from scipy.special import softmax

DPP = '/home/communications/workdir/doppelgangers-plusplus'
sys.path.insert(0, DPP)
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs

ap = argparse.ArgumentParser()
ap.add_argument('db'); ap.add_argument('images')
ap.add_argument('--threshold', type=float, default=0.8)
ap.add_argument('--window', type=int, default=10)
ap.add_argument('--block', type=int, default=10)
a = ap.parse_args()

db = sqlite3.connect(a.db)
names = dict(db.execute('select image_id, name from images'))
walk = {i: k for k, i in enumerate(sorted(names, key=lambda i: names[i]))}
blocks = collections.defaultdict(list)          # (block_a, block_b) -> [(inliers, pair_id, img_a, img_b)]
n_all = n_near = 0
for pid, m in db.execute('select pair_id, rows from two_view_geometries where rows > 0'):
    n_all += 1
    j = pid % 2147483647; i = (pid - j) // 2147483647
    wa, wb = sorted((walk[i], walk[j]))
    if wb - wa <= a.window:
        n_near += 1; continue
    blocks[(wa // a.block, wb // a.block)].append((m, pid, i, j))
print(f'dg_filter: {n_all} verified pairs; {n_near} walk neighbours kept unscored; '
      f'{n_all - n_near} others in {len(blocks)} blocks of {a.block}x{a.block} photos', flush=True)

model = AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt',
                         head_type_dg='transformer', output_mode='pts3d+desc24', output_mode_dg='dg_score',
                         depth_mode=('exp', -np.inf, np.inf), conf_mode=('exp', 1, np.inf), enc_embed_dim=1024, enc_depth=24,
                         enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True,
                         desc_conf_mode=('exp', 0, np.inf), add_dg_pred_head=True,
                         freeze=['mask', 'encoder', 'decoder', 'head']).from_pretrained(f'{DPP}/checkpoints/checkpoint-dg+visym.pth').to('cuda')


def score(na, nb):
    """D++'s own vote over its two heads, both orders (colmap_usage.py)."""
    out = inference(make_pairs(load_images([f'{a.images}/{na}', f'{a.images}/{nb}'], size=512, verbose=False)), model, 'cuda', verbose=False)
    p = [torch.stack(x) if isinstance(x, list) else x for x in (out['pred1'], out['pred2'])]
    s1, s2 = (softmax(x.detach().cpu().numpy(), axis=1) for x in p)
    v0 = sum(s1[:, 0] > s1[:, 1]) + sum(s2[:, 0] > s2[:, 1]); v1 = sum(s1[:, 1] > s1[:, 0]) + sum(s2[:, 1] > s2[:, 0])
    return float(np.max((s1[:, 1], s2[:, 1])) if v1 > v0 else np.min((s1[:, 1], s2[:, 1])) if v1 < v0 else np.mean((s1[:, 1], s2[:, 1])))


t0, drop, n_blocks_drop, strong_drop = time.time(), [], 0, 0
for k, (key, prs) in enumerate(sorted(blocks.items())):
    m, pid, i, j = max(prs)
    if score(names[i], names[j]) < a.threshold:
        n_blocks_drop += 1; drop += [p[1] for p in prs]; strong_drop += sum(p[0] >= 150 for p in prs)
    if (k + 1) % 500 == 0:
        print(f'dg_filter: {k + 1}/{len(blocks)} blocks, {n_blocks_drop} look-alike, {(time.time() - t0) / (k + 1):.2f} s/block', flush=True)
for table in ('matches', 'two_view_geometries'):
    db.executemany(f'delete from {table} where pair_id = ?', [(p,) for p in drop])
db.commit()
print(f'dg_filter: {n_blocks_drop}/{len(blocks)} blocks judged look-alike (< {a.threshold}); deleted {len(drop)} pairs '
      f'({strong_drop} with >=150 inliers) in {(time.time() - t0) / 60:.1f} min')
