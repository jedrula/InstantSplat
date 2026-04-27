#!/usr/bin/env python3
"""
select_frames.py — High-quality keyframe selection for 3DGS/SfM pipelines.

Takes a video, oversamples it, then greedily selects the best subset for
reconstruction — filtering blurry frames, avoiding duplicates, and ensuring
good temporal coverage.

Usage:
    python select_frames.py video.mp4
    python select_frames.py video.mp4 --fps 5 --target 60 --out ./out
    python select_frames.py video.mp4 --fps 3 --target 20 --start 5 --duration 30
    python select_frames.py video.mp4 --no-extract   # re-run selection on existing frames

Outputs:
    <out>/all/          all extracted frames (PNG)
    <out>/selected/     the chosen keyframes (copied, ready for MASt3R / COLMAP)
    <out>/thumbs/       thumbnails used by debug.html
    <out>/debug.html    visual debug page — open in browser
    <out>/metadata.json full per-frame data (blur, score, status, …)

TODO — upgrade path:
    - Replace ORB with SuperPoint / MASt3R features for better matches on
      textureless or repetitive surfaces (climbing walls, plain rock).
    - Replace ratio-test match count with RANSAC inlier count (geometric
      verification) to eliminate false matches from repetitive patterns.
    - Add global coverage constraint: cluster frames by appearance and enforce
      minimum selection from each cluster to avoid gaps across the whole scene.
    - Add temporal smoothness: prefer selections that are evenly spaced in time,
      not just by information gain.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# ── Defaults ─────────────────────────────────────────────────────────────────

THUMB_W = 180          # px — thumbnail width in debug HTML grid
TIMELINE_BAR_W = 3     # px — width of each frame bar in the timeline strip
DEFAULT_FPS = 3.0
DEFAULT_TARGET = 60
DEFAULT_NEIGHBORS = 5
DEFAULT_MIN_MATCHES = 40
DEFAULT_MAX_MATCHES = 400
DEFAULT_BLUR_PCTILE = 10.0   # hard-reject frames below this blur percentile


# ── Feature extraction & matching ────────────────────────────────────────────

def compute_blur(img: np.ndarray) -> float:
    """
    95th-percentile of absolute Laplacian values.

    Why not variance: on a climbing wall most of the frame is plain beige
    background — few edges, low variance — making sharp background frames
    look blurry. Taking the high percentile instead finds the sharpest
    edges anywhere in the frame (holds, chalk marks, wall features) and
    ignores the empty regions. A truly blurry frame has no sharp edges
    anywhere, so the 95th percentile stays low regardless of content.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.percentile(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), 95))


def extract_orb(img: np.ndarray, orb) -> tuple:
    """
    Extract ORB keypoints and descriptors.

    ORB is fast and runs on CPU — good baseline for a preprocessing step.
    TODO: replace with SuperPoint (GPU) or MASt3R features for better quality
    on repetitive textures (e.g. climbing wall beige background).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(gray, None)


def count_matches(des1, des2, bf) -> int:
    """
    Lowe's ratio test match count between two descriptor sets.
    Returns a proxy for frame-to-frame similarity.

    TODO: replace with RANSAC inlier count (essential matrix) to eliminate
    false matches caused by repetitive patterns — more expensive but much
    more reliable as a "do these frames actually overlap?" signal.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return 0
    return sum(1 for m, n in matches if m.distance < 0.75 * n.distance)


# ── Core selection algorithm ──────────────────────────────────────────────────

def select_frames(
    paths: list,
    target: int,
    neighbors: int,
    min_matches: int,
    max_matches: int,
    w_sharp: float = 1.0,
    w_conn: float = 1.0,
    w_redundancy: float = 1.0,
    blur_pctile_cutoff: float = DEFAULT_BLUR_PCTILE,
    min_gap: int = -1,
) -> tuple:
    """
    Select `target` high-quality, evenly-distributed frames from `paths`.

    Returns:
        selected    list of selected indices (in temporal order)
        frame_data  list of per-frame dicts (for debug HTML / metadata.json)

    Algorithm:
        1. Compute blur score (95th-pctile abs-Laplacian) for every frame.
        2. Divide the timeline into `target` equal-width bins.
        3. Within each bin, pick the sharpest non-blurry frame that is at
           least `min_gap` frames from the previous selection.
           If no candidate satisfies the gap, fall back to any frame in the bin.
           If every frame in a bin is below the blur floor, pick the least blurry.
        4. Build a local ORB similarity graph for the HTML visualisation
           (shows overlap between consecutive selected frames — not a selection gate).

    min_gap=-1 means auto: half the bin size, minimum 2.

    Why bins instead of greedy:
        Greedy chases information gain and clusters selections around textured
        regions (dense holds), leaving long plain sections unrepresented.
        Equal bins guarantee one frame per time-slice of the clip regardless
        of local texture density.
    """
    N = len(paths)

    # ── Load images ──────────────────────────────────────────────────────────
    print("  Loading images…")
    images = []
    for p in tqdm(paths, leave=False):
        img = cv2.imread(p)
        if img is None:
            sys.exit(f"Could not read image: {p}")
        images.append(img)

    # ── Blur scores ───────────────────────────────────────────────────────────
    print("  Computing blur scores…")
    blur_raw = np.array([compute_blur(img) for img in tqdm(images, leave=False)])
    blur_norm = blur_raw / (blur_raw.max() + 1e-6)
    hard_blur_floor = np.percentile(blur_raw, blur_pctile_cutoff)

    # ── Bin-based selection ───────────────────────────────────────────────────
    # Divide [0, N) into `target` equal bins; pick sharpest frame in each.
    bin_size = N / target
    _gap = max(2, int(bin_size // 2)) if min_gap < 0 else min_gap
    selected = []
    last = -_gap  # tracks index of the most recently selected frame
    for b in range(target):
        lo = int(b * bin_size)
        hi = min(int((b + 1) * bin_size), N)
        if lo >= hi:
            continue
        # Prefer candidates that are at least _gap frames from last selection.
        gapped = [i for i in range(lo, hi) if i - last >= _gap]
        pool_all = gapped if gapped else list(range(lo, hi))
        non_blurry = [i for i in pool_all if blur_raw[i] >= hard_blur_floor]
        pool = non_blurry if non_blurry else pool_all
        best = max(pool, key=lambda i: blur_raw[i])
        selected.append(best)
        last = best

    # ── ORB similarity graph (display only) ──────────────────────────────────
    # Computed for consecutive selected-frame pairs so the HTML can show
    # how much overlap neighbouring keyframes share.
    print("  Extracting ORB features for overlap display…")
    orb = cv2.ORB_create(5000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    features = [extract_orb(images[i], orb) for i in tqdm(selected, leave=False)]
    sel_overlap = {}  # sel_overlap[i] = matches between selected[i] and selected[i-1]
    for k in range(1, len(selected)):
        m = count_matches(features[k - 1][1], features[k][1], bf)
        sel_overlap[selected[k]] = m

    # ── Build per-frame metadata ──────────────────────────────────────────────
    selected_set = set(selected)
    sel_order    = {idx: order for order, idx in enumerate(selected)}
    frame_data   = []
    for i in range(N):
        b = int(i / bin_size)
        if i in selected_set:
            status = 'selected'
        elif blur_raw[i] < hard_blur_floor:
            status = 'blurry'
        else:
            status = 'skipped'

        frame_data.append({
            'idx':             i,
            'path':            paths[i],
            'blur_raw':        float(blur_raw[i]),
            'blur_norm':       float(blur_norm[i]),
            'score':           float(blur_norm[i]),   # score == sharpness in bin model
            'conn':            0.0,
            'redundancy':      0.0,
            'status':          status,
            'selection_order': sel_order.get(i),
            'bin':             b,
            'overlap_prev':    sel_overlap.get(i),    # None for non-selected or first
        })

    return selected, frame_data


# ── Thumbnails ────────────────────────────────────────────────────────────────

def make_thumb(src: str, dst: str, width: int = THUMB_W):
    img = cv2.imread(src)
    if img is None:
        return
    h, w = img.shape[:2]
    thumb = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])


# ── HTML generation ───────────────────────────────────────────────────────────

_STATUS_COLOR = {
    'selected':   '#4caf50',
    'blurry':     '#f44336',
    'no_overlap': '#ff9800',
    'skipped':    '#555555',
}
_STATUS_LABEL = {
    'selected':   'Selected',
    'blurry':     'Rejected — blurry',
    'no_overlap': 'No overlap',
    'skipped':    'Skipped',
}


def generate_html(frame_data: list, selected: list, out_dir: Path, args) -> str:
    N = len(frame_data)
    n_sel = len(selected)
    n_blurry = sum(1 for f in frame_data if f['status'] == 'blurry')
    n_noop = sum(1 for f in frame_data if f['status'] == 'no_overlap')
    n_skip = N - n_sel - n_blurry - n_noop

    # Timeline bars
    tl_parts = []
    for f in frame_data:
        color = _STATUS_COLOR.get(f['status'], '#555')
        label = _STATUS_LABEL.get(f['status'], '')
        idx   = f['idx']
        tl_parts.append(
            f'<div class="tbar" style="background:{color}" '
            f'title="#{idx:04d} {label}" onclick="scrollTo({idx})"></div>'
        )
    tl_bars = '\n'.join(tl_parts)

    # Frame cards
    cards = []
    for f in frame_data:
        thumb_rel = f"thumbs/frame_{f['idx']:04d}_thumb.jpg"
        full_rel  = f"all/frame_{f['idx']:04d}.png"
        img_tag = (
            f'<img src="{thumb_rel}" loading="lazy">'
            if (out_dir / thumb_rel).exists()
            else '<div class="no-thumb">—</div>'
        )
        color = _STATUS_COLOR.get(f['status'], '#555')
        label = _STATUS_LABEL.get(f['status'], f['status'])
        order = (f'<span class="order">#{f["selection_order"]+1}</span>'
                 if f['selection_order'] is not None else '')
        blur_pct = int(f['blur_norm'] * 100)
        cards.append(f'''
<div class="card {f["status"]}" id="f{f['idx']:04d}" data-status="{f['status']}"
     data-full="{full_rel}" data-idx="{f['idx']}"
     onclick="openLightbox({f['idx']})">
  {img_tag}
  <div class="meta">
    <div class="row"><span class="idx">frame {f['idx']:04d}</span>{order}</div>
    <div class="row"><span class="badge" style="background:{color}">{label}</span></div>
    <div class="row dim">blur {blur_pct}% · score {f["score"]:.2f}</div>
  </div>
</div>''')

    cards_html = '\n'.join(cards)
    params = (f"fps={args.fps} · target={args.target} · neighbors={args.neighbors} · "
              f"min_matches={args.min_matches} · max_matches={args.max_matches} · "
              f"blur_pctile={args.blur_pctile}")

    # JSON blobs embedded in the page for the lightbox JS
    import json as _json
    frames_json      = _json.dumps(frame_data)
    status_color_json = _json.dumps(_STATUS_COLOR)
    status_label_json = _json.dumps(_STATUS_LABEL)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Frame Selection Debug</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#111;color:#ddd;font-family:monospace;font-size:13px}}
a{{color:#90caf9}}

header{{padding:14px 18px;background:#1b1b1b;border-bottom:1px solid #2a2a2a}}
h1{{font-size:15px;font-weight:bold;color:#fff;margin-bottom:10px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px}}
.stat{{padding:3px 10px;border-radius:3px;font-size:12px}}
.s-total{{background:#263238;color:#90a4ae}}
.s-selected{{background:#1b5e20;color:#a5d6a7}}
.s-blurry{{background:#7f0000;color:#ffcdd2}}
.s-noop{{background:#bf360c;color:#ffe0b2}}
.s-skip{{background:#333;color:#999}}
.params{{font-size:11px;color:#666;margin-top:4px}}

.filters{{padding:8px 18px;background:#181818;border-bottom:1px solid #2a2a2a;display:flex;gap:8px;align-items:center}}
.filters span{{color:#777;font-size:12px}}
button{{background:#2a2a2a;color:#bbb;border:1px solid #444;padding:3px 11px;border-radius:3px;cursor:pointer;font-family:monospace;font-size:12px}}
button:hover{{background:#383838}}
button.active{{background:#2e7d32;color:#fff;border-color:#4caf50}}

.tl-wrap{{padding:8px 18px 10px;background:#161616;border-bottom:1px solid #222}}
.tl-label{{font-size:11px;color:#555;margin-bottom:5px}}
.timeline{{display:flex;gap:1px;height:36px;overflow-x:auto}}
.tbar{{flex:0 0 {TIMELINE_BAR_W}px;height:100%;cursor:pointer;opacity:.8;transition:opacity .1s}}
.tbar:hover{{opacity:1}}

.legend{{padding:5px 18px;background:#161616;border-bottom:1px solid #222;display:flex;gap:16px;font-size:11px;color:#888;flex-wrap:wrap}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:3px;vertical-align:middle}}

.grid{{display:flex;flex-wrap:wrap;gap:8px;padding:14px 18px}}
.card{{width:{THUMB_W}px;border:2px solid #2a2a2a;border-radius:4px;overflow:hidden;background:#181818}}
.card img{{width:100%;display:block;cursor:pointer}}
.card{{cursor:pointer}}
.card:hover img{{opacity:.85}}
.card.selected{{border-color:#4caf50}}
.card.blurry{{border-color:#f44336;opacity:.35}}
.card.no_overlap{{border-color:#ff9800;opacity:.4}}
.card.skipped{{opacity:.22}}
.card.hidden{{display:none}}
.no-thumb{{height:70px;display:flex;align-items:center;justify-content:center;color:#444}}
.meta{{padding:5px 7px;display:flex;flex-direction:column;gap:3px;font-size:10px}}
.row{{display:flex;align-items:center;gap:5px;flex-wrap:wrap}}
.idx{{color:#777}}
.order{{color:#81c784;font-weight:bold}}
.badge{{padding:1px 5px;border-radius:2px;color:#fff;font-size:10px}}
.dim{{color:#556}}

/* lightbox */
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:999;flex-direction:column;align-items:center;justify-content:center}}
#lb.open{{display:flex}}
#lb img{{max-width:95vw;max-height:85vh;object-fit:contain;border:1px solid #333}}
#lb-meta{{margin-top:10px;font-size:12px;color:#aaa;text-align:center}}
#lb-meta .badge{{padding:2px 7px;border-radius:3px;color:#fff;font-size:11px;margin-left:6px}}
#lb-nav{{position:fixed;top:50%;transform:translateY(-50%);display:flex;justify-content:space-between;width:100%;padding:0 12px;pointer-events:none}}
.lb-btn{{pointer-events:all;background:rgba(255,255,255,.1);border:none;color:#fff;font-size:28px;padding:10px 18px;cursor:pointer;border-radius:4px;line-height:1}}
.lb-btn:hover{{background:rgba(255,255,255,.25)}}
#lb-close{{position:fixed;top:14px;right:18px;background:none;border:none;color:#aaa;font-size:24px;cursor:pointer;line-height:1}}
#lb-close:hover{{color:#fff}}
</style>
</head>
<body>

<header>
  <h1>Frame Selection Debug</h1>
  <div class="stats">
    <div class="stat s-total">Total {N}</div>
    <div class="stat s-selected">✓ Selected {n_sel}</div>
    <div class="stat s-blurry">✗ Blurry {n_blurry}</div>
    <div class="stat s-noop">⚠ No overlap {n_noop}</div>
    <div class="stat s-skip">· Skipped {n_skip}</div>
  </div>
  <div class="params">{params}</div>
</header>

<div class="filters">
  <span>Show:</span>
  <button class="active" onclick="filter('all',this)">All</button>
  <button onclick="filter('selected',this)">Selected</button>
  <button onclick="filter('blurry',this)">Blurry</button>
  <button onclick="filter('no_overlap',this)">No overlap</button>
  <button onclick="filter('skipped',this)">Skipped</button>
</div>

<div class="tl-wrap">
  <div class="tl-label">Timeline — one bar per frame, click to jump</div>
  <div class="timeline">
{tl_bars}
  </div>
</div>

<div class="legend">
  <span><span class="dot" style="background:#4caf50"></span>Selected</span>
  <span><span class="dot" style="background:#f44336"></span>Blurry (hard reject below {args.blur_pctile:.0f}th pctile)</span>
  <span><span class="dot" style="background:#ff9800"></span>No overlap — possible coverage gap</span>
  <span><span class="dot" style="background:#555"></span>Skipped (redundant)</span>
</div>

<div class="grid" id="grid">
{cards_html}
</div>

<!-- lightbox -->
<div id="lb" onclick="closeLightbox(event)">
  <button id="lb-close" onclick="closeLightbox()">✕</button>
  <img id="lb-img" src="" alt="">
  <div id="lb-meta"></div>
  <div id="lb-nav">
    <button class="lb-btn" onclick="event.stopPropagation();stepLightbox(-1)">&#8592;</button>
    <button class="lb-btn" onclick="event.stopPropagation();stepLightbox(1)">&#8594;</button>
  </div>
</div>

<script>
const FRAMES = {frames_json};

function filter(s, btn) {{
  document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.card').forEach(c => {{
    c.classList.toggle('hidden', s !== 'all' && c.dataset.status !== s);
  }});
}}

function scrollTo(idx) {{
  const el = document.getElementById('f' + String(idx).padStart(4,'0'));
  if (!el) return;
  if (el.classList.contains('hidden')) filter('all', document.querySelector('button'));
  el.scrollIntoView({{behavior:'smooth', block:'center'}});
  el.style.outline = '3px solid #fff';
  setTimeout(() => el.style.outline = '', 1600);
}}

let lbIdx = 0;
const STATUS_COLOR = {status_color_json};
const STATUS_LABEL = {status_label_json};

function openLightbox(idx) {{
  lbIdx = idx;
  showFrame(idx);
  document.getElementById('lb').classList.add('open');
}}

function showFrame(idx) {{
  const f = FRAMES[idx];
  document.getElementById('lb-img').src = 'all/frame_' + String(idx).padStart(4,'0') + '.png';
  const color = STATUS_COLOR[f.status] || '#555';
  const label = STATUS_LABEL[f.status] || f.status;
  const order = f.selection_order !== null ? ` &nbsp;#${{f.selection_order+1}}` : '';
  document.getElementById('lb-meta').innerHTML =
    `frame ${{String(idx).padStart(4,'0')}}${{order}} &nbsp;`+
    `<span class="badge" style="background:${{color}}">${{label}}</span> &nbsp;`+
    `blur ${{Math.round(f.blur_norm*100)}}% &nbsp; score ${{f.score.toFixed(2)}}`;
  lbIdx = idx;
}}

function stepLightbox(dir) {{
  let next = lbIdx + dir;
  if (next < 0) next = FRAMES.length - 1;
  if (next >= FRAMES.length) next = 0;
  showFrame(next);
}}

function closeLightbox(e) {{
  if (e && e.target !== document.getElementById('lb') && e.target !== document.getElementById('lb-close')) return;
  document.getElementById('lb').classList.remove('open');
}}

document.addEventListener('keydown', e => {{
  if (!document.getElementById('lb').classList.contains('open')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowRight') stepLightbox(1);
  if (e.key === 'ArrowLeft')  stepLightbox(-1);
}});
</script>

</body>
</html>'''


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video: str, out_dir: Path, fps: float,
                   start: float, duration: Optional[float]) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(start), '-i', video]
    if duration is not None:
        cmd += ['-t', str(duration)]
    cmd += ['-vf', f'fps={fps}', str(out_dir / 'frame_%04d.png')]
    subprocess.run(cmd, check=True)
    return sorted(str(p) for p in out_dir.glob('frame_*.png'))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('video', help='Input video file')
    p.add_argument('--out', default='frame_selection',
                   help='Output directory (default: frame_selection/)')
    p.add_argument('--fps', type=float, default=DEFAULT_FPS,
                   help=f'Extraction FPS for oversampling (default: {DEFAULT_FPS})')
    p.add_argument('--target', type=int, default=DEFAULT_TARGET,
                   help=f'Target number of selected frames (default: {DEFAULT_TARGET})')
    p.add_argument('--start', type=float, default=0.0,
                   help='Start time in seconds (default: 0)')
    p.add_argument('--duration', type=float, default=None,
                   help='Duration in seconds to process (default: full video)')
    p.add_argument('--neighbors', type=int, default=DEFAULT_NEIGHBORS,
                   help=f'Temporal window for similarity graph (default: {DEFAULT_NEIGHBORS})')
    p.add_argument('--min-matches', type=int, default=DEFAULT_MIN_MATCHES,
                   dest='min_matches',
                   help=f'Min ORB matches required for overlap (default: {DEFAULT_MIN_MATCHES})')
    p.add_argument('--max-matches', type=int, default=DEFAULT_MAX_MATCHES,
                   dest='max_matches',
                   help=f'Match count above which frames are considered redundant (default: {DEFAULT_MAX_MATCHES})')
    p.add_argument('--blur-pctile', type=float, default=DEFAULT_BLUR_PCTILE,
                   dest='blur_pctile',
                   help=f'Hard-reject frames below this blur percentile (default: {DEFAULT_BLUR_PCTILE})')
    p.add_argument('--min-gap', type=int, default=-1,
                   dest='min_gap',
                   help='Minimum frame index gap between consecutive selections. '
                        '-1 = auto (half bin size, at least 2). 0 = disabled.')
    p.add_argument('--no-extract', action='store_true',
                   help='Skip ffmpeg extraction — re-run selection on existing <out>/all/ frames')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir  = Path(args.out)
    all_dir  = out_dir / 'all'
    sel_dir  = out_dir / 'selected'
    thumb_dir = out_dir / 'thumbs'

    # ── 1. Extract ────────────────────────────────────────────────────────────
    if not args.no_extract:
        print(f"[1/5] Extracting frames at {args.fps} fps…")
        paths = extract_frames(args.video, all_dir, args.fps, args.start, args.duration)
    else:
        paths = sorted(str(p) for p in all_dir.glob('frame_*.png'))
        print(f"[1/5] Using {len(paths)} existing frames from {all_dir}")

    if not paths:
        sys.exit("No frames found.")
    print(f"      {len(paths)} frames")

    # ── 2. Select ─────────────────────────────────────────────────────────────
    print(f"[2/5] Running selection (target={args.target})…")
    selected, frame_data = select_frames(
        paths,
        target=args.target,
        neighbors=args.neighbors,
        min_matches=args.min_matches,
        max_matches=args.max_matches,
        blur_pctile_cutoff=args.blur_pctile,
        min_gap=args.min_gap,
    )
    n_sel = len(selected)
    n_blurry = sum(1 for f in frame_data if f['status'] == 'blurry')
    n_noop   = sum(1 for f in frame_data if f['status'] == 'no_overlap')
    print(f"      selected={n_sel}  blurry={n_blurry}  no_overlap={n_noop}  skipped={len(paths)-n_sel-n_blurry-n_noop}")

    # ── 3. Copy selected frames ───────────────────────────────────────────────
    print(f"[3/5] Copying selected frames → {sel_dir}")
    if sel_dir.exists():
        shutil.rmtree(sel_dir)
    sel_dir.mkdir(parents=True)
    for fd in frame_data:
        if fd['status'] == 'selected':
            shutil.copy2(fd['path'], sel_dir / Path(fd['path']).name)

    # ── 4. Thumbnails ─────────────────────────────────────────────────────────
    print(f"[4/5] Generating thumbnails…")
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for fd in tqdm(frame_data, leave=False):
        dst = thumb_dir / f"frame_{fd['idx']:04d}_thumb.jpg"
        if not dst.exists():
            make_thumb(fd['path'], str(dst))

    # ── 5. Debug HTML + metadata ──────────────────────────────────────────────
    print(f"[5/5] Writing debug.html + metadata.json…")
    html_path = out_dir / 'debug.html'
    html_path.write_text(generate_html(frame_data, selected, out_dir, args))

    (out_dir / 'metadata.json').write_text(json.dumps({
        'video':            args.video,
        'fps':              args.fps,
        'target':           args.target,
        'n_total':          len(paths),
        'n_selected':       n_sel,
        'selected_indices': selected,
        'frames':           frame_data,
    }, indent=2))

    print()
    print("╔══════════════════════════════════════════════════╗")
    print(f"║  Done!  {n_sel}/{len(paths)} frames selected")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Selected  → {sel_dir}")
    print(f"║  Debug     → {html_path}")
    print("╚══════════════════════════════════════════════════╝")


if __name__ == '__main__':
    main()
