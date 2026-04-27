# Pipeline Comparison: InstantSplat vs GLOMAP+gsplat

## Summary

For our use case (short climbing-wall videos, 6–15 frames, near-planar scenes),
**InstantSplat (MASt3R + vanilla 3DGS) wins by a large margin.**
The GLOMAP+gsplat approach produced floating Gaussians with no recognizable structure.

---

## What we tried

### Pipeline A — InstantSplat (`video_to_splat.sh`)

```
video → frames → MASt3R dense feature matching → global alignment → 3DGS training → .splat
```

- MASt3R uses learned dense features (not SIFT), so repetitive textures (beige wall backgrounds)
  don't cause false matches.
- Global alignment gives dense, metric-scale point clouds: ~1.47M init points for 10 frames.
- 6–10 frames on `fpinka.mp4`: produces a recognizable, well-structured splat of the climbing wall.
- OOM limit on our 8GB GPU (with ~1.25GB used by other services): 15 frames hits OOM during
  MASt3R global alignment. True ceiling is somewhere between 10 and 15 frames when the GPU is
  fully free.

### Pipeline B — GLOMAP+gsplat (`video_to_splat_glomap.sh`)

```
video → frames → COLMAP SIFT features → COLMAP exhaustive/sequential matching
      → COLMAP incremental mapper (originally GLOMAP) → outlier filter → gsplat training → .splat
```

Results: floating Gaussians, no structure. Never produced a usable splat.

---

## Why Pipeline B failed

### 1. SIFT on repetitive texture

The climbing wall has a large beige background with sparse, visually similar holds.
SIFT produces many false matches. For a 17-frame run, only 37 of 190 pairs matched
(and those 37 were low-inlier matches). Most frames were geometrically isolated.

### 2. Near-planar scene breaks global SfM (GLOMAP)

GLOMAP uses rotation averaging + global position estimation. On near-planar scenes,
the essential matrix decomposition is ambiguous, and the global positioning step fails
silently — assigning one camera a position ~12,000 units away from all others
(e.g. frame 5 at [3528, −2202, 12599] when all others were within ±1 unit of origin).
This poisoned all 3DGS training.

We switched to COLMAP incremental mapper, which is frame-by-frame and detects
degeneracies, but the upstream SIFT matching failure meant most frames were still
disconnected.

### 3. Some frames are geometrically incompatible

Frames 1–4 of `fpinka.mp4` (at 00:01 start) are close-ups of large colored holds and
a person/mat, at a completely different scale and section of the wall from frames 5–17.
No SIFT inliers connect them to the rest. Removing them helped but wasn't sufficient.

### 4. Sparse init point cloud

Even in the best case, COLMAP/GLOMAP gave a very sparse point cloud from SIFT matches.
InstantSplat's MASt3R produces dense per-pixel predictions, giving a much better
starting point for 3DGS optimization.

---

## Way forward

**Keep using InstantSplat.** The key levers for better quality:

1. **More frames** — quality improves with more viewpoints. Current GPU ceiling is ~10–14
   frames (depending on what else is running). Options:
   - Kill other GPU processes before a run to reclaim ~1.25 GB
   - Switch to Fast3R (feed-forward, no O(N²) alignment step) — see TODO.md
   - Run on a machine with more VRAM

2. **Better frame selection** — not all frames are equal. Prefer:
   - Frames spread across the full wall section (not close-ups or transitions)
   - Good overlap between adjacent frames (~60–80% shared area)
   - Sharp, well-lit frames (avoid motion blur from fast pans)
   - Avoid frames where the camera is very close (different scale from the rest)

3. **Multiple video clips** — `video_to_splat.sh` already supports multiple input videos.
   Filming the same section from slightly different angles (or a second pass) adds
   viewpoint diversity without needing a longer clip.

4. **Start time** — `--start T` skips the first T seconds. Use it to skip any close-up
   intro or non-wall footage before the steady walkthrough begins.

---

## Files from Pipeline B (kept for reference)

- `video_to_splat_glomap.sh` — full GLOMAP+gsplat pipeline script
- `filter_sfm_outliers.py` — detects and removes cameras >3σ from centroid
- `gsplat_examples/` — gsplat simple_trainer integration
  - `datasets/colmap.py` — pycolmap 4.0.3 compatibility shim (`SceneManager` removed)
  - `datasets/normalize.py` — camera normalization utilities
- `simple_trainer.py` — gsplat trainer (viewer imports wrapped for headless use)
- `gsplat_utils.py` — renamed from `utils.py` to avoid shadowing InstantSplat's `utils/` package
