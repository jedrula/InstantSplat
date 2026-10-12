#!/usr/bin/env python3
"""Even out per-photo exposure / white balance BEFORE training, so Brush need not grow fog to explain it.

U2's photos run ISO 400-1250 with auto white balance (no exposure lock), and the unions show near-camera haze that
photo-support pruning cannot remove: it is used by many views, i.e. it explains something every view shares in
part -- like a brightness that changes frame to frame. Brush has no per-photo appearance model, so fix the input:
every SfM point should have ONE colour; solve per photo and channel a gain g_i (linear light) with
  log c(point p in photo i) = log albedo_p + g_i
by alternating medians (robust to specular holds and occlusion), centre the gains (median 0), and divide each photo
by exp(g_i). Saturated and near-black samples are ignored. EXIF is kept.

Pipeline step: video_to_splat.sh --normalize-exposure (Brush), writing $MODEL_DIR/expnorm; the pod's photos stay untouched.

usage: normalize_exposure.py SRC_DATASET OUT_DATASET      (dataset = images/ + sparse/0 that pair)
"""
import os, shutil, sys
import numpy as np, pycolmap
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
rec = pycolmap.Reconstruction(f'{src}/sparse/0')
lin = lambda x: np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
srgb = lambda x: np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)

ims = sorted(rec.images.values(), key=lambda im: im.name)
idx = {im.image_id: k for k, im in enumerate(ims)}
obs_i, obs_p, obs_c = [], [], []
for im in ims:
    a = np.asarray(Image.open(f'{src}/images/{im.name}').convert('RGB')).astype(np.float32) / 255
    H, W = a.shape[:2]
    pts = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
    if not pts:
        continue
    xy = np.array([q[0] for q in pts]); pid = np.array([q[1] for q in pts])
    x = np.clip(np.round(xy[:, 0]).astype(int), 1, W - 2); y = np.clip(np.round(xy[:, 1]).astype(int), 1, H - 2)
    c = np.mean([a[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)       # 3x3 mean
    ok = (c.max(1) < 0.97) & (c.min(1) > 0.03)
    obs_i.append(np.full(ok.sum(), idx[im.image_id])); obs_p.append(pid[ok]); obs_c.append(lin(c[ok]))
oi, op_, oc = np.concatenate(obs_i), np.concatenate(obs_p), np.log(np.concatenate(obs_c))
_, pk = np.unique(op_, return_inverse=True)
n_img, n_pt = len(ims), pk.max() + 1
order = np.argsort(oi, kind='stable'); oi, pk, oc = oi[order], pk[order], oc[order]
cuts = np.searchsorted(oi, np.arange(n_img + 1))                  # photo i's observations are oi[cuts[i]:cuts[i+1]]
cnt = np.bincount(pk, minlength=n_pt)[:, None]
g = np.zeros((n_img, 3))
for _ in range(15):
    alb = np.zeros((n_pt, 3)); np.add.at(alb, pk, oc - g[oi]); alb /= np.maximum(cnt, 1)
    r = oc - alb[pk]
    g = np.stack([np.median(r[cuts[i]:cuts[i + 1]], 0) if cuts[i + 1] > cuts[i] else np.zeros(3) for i in range(n_img)])
    g -= np.median(g, 0)
stops = g.mean(1) / np.log(2)
print(f'{len(oc):,} observations of {n_pt:,} points; gain spread (stops, mean of RGB): '
      f'p5 {np.percentile(stops, 5):+.2f} p50 {np.median(stops):+.2f} p95 {np.percentile(stops, 95):+.2f}; '
      f'channel spread (WB) p95 {np.percentile(np.ptp(g, 1) / np.log(2), 95):.2f} stops', flush=True)

os.makedirs(f'{out}/images', exist_ok=True)
os.symlink(os.path.realpath(f'{src}/sparse'), f'{out}/sparse')          # same poses, only the pixels change
for k, im in enumerate(ims):
    p = Image.open(f'{src}/images/{im.name}')
    a = lin(np.asarray(p.convert('RGB')).astype(np.float32) / 255) * np.exp(-g[k])
    Image.fromarray((np.clip(srgb(a), 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        f'{out}/images/{im.name}', quality=95, exif=p.info.get('exif', b''))
np.savetxt(f'{out}/gains_log.txt', g, header='per-photo log gain R G B (divided out), sorted by name')
print('wrote', out, flush=True)
