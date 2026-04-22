#!/bin/bash
# video_to_splat.sh — full pipeline: video → frames → 3DGS → .splat
#
# Usage:
#   bash video_to_splat.sh <video> [options]
#
# Options:
#   --scene NAME     scene name / output folder (default: video filename stem)
#   --n-frames N     number of frames to extract (default: 6)
#   --start T        start time in video, e.g. 00:00:05 (default: 0)
#   --iters N        training iterations (default: 1000)
#
# Example:
#   bash video_to_splat.sh /path/to/fpinka.mp4 --n-frames 6 --start 00:00:01 --iters 1000

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/communications/miniconda3/envs/instantsplat/bin/python

# ── Defaults ─────────────────────────────────────────────────────────────────
N_FRAMES=6
START=0
ITERS=1000
SCENE=""
VIDEO=""

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)     SCENE="$2";    shift 2 ;;
        --n-frames)  N_FRAMES="$2"; shift 2 ;;
        --start)     START="$2";    shift 2 ;;
        --iters)     ITERS="$2";    shift 2 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           VIDEO="$1";    shift ;;
    esac
done

if [[ -z "$VIDEO" ]]; then
    echo "Usage: bash video_to_splat.sh <video> [--scene NAME] [--n-frames N] [--start T] [--iters N]"
    exit 1
fi

if [[ ! -f "$VIDEO" ]]; then
    echo "Error: video not found: $VIDEO"
    exit 1
fi

if [[ -z "$SCENE" ]]; then
    SCENE="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
fi

SCENE_DIR="$REPO/assets/examples/$SCENE"
IMAGE_DIR="$SCENE_DIR/images"
MODEL_DIR="$REPO/output_infer/$SCENE"
PLY="$MODEL_DIR/point_cloud/iteration_${ITERS}/point_cloud.ply"
SPLAT_OUT="$MODEL_DIR/$SCENE.splat"

# ── Step 1: extract frames ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  video_to_splat: $SCENE"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "[1/3] Extracting $N_FRAMES frames (start=$START)..."

rm -rf "$IMAGE_DIR"
mkdir -p "$IMAGE_DIR"

ffmpeg -y -loglevel error \
    -ss "$START" \
    -i "$VIDEO" \
    -vf fps=1 \
    -frames:v "$N_FRAMES" \
    "$IMAGE_DIR/frame_%04d.jpg"

echo "    → $(ls "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l) frames → $IMAGE_DIR"

# ── Step 2: geometry init + 3DGS training ────────────────────────────────────
echo ""
echo "[2/3] MASt3R init + 3DGS training ($ITERS iterations)..."
mkdir -p "$MODEL_DIR"
cd "$REPO"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -W ignore ./init_geo.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    --n_views "$N_FRAMES" \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    --infer_video \
    2>&1 | tee "$MODEL_DIR/01_init_geo.log"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" ./train.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    -r 1 \
    --n_views "$N_FRAMES" \
    --iterations "$ITERS" \
    --pp_optimizer \
    --optim_pose \
    2>&1 | tee "$MODEL_DIR/02_train.log"

echo "    → done"

# ── Step 3: PLY → .splat ─────────────────────────────────────────────────────
echo ""
echo "[3/3] Converting to .splat..."
"$PYTHON" "$REPO/ply2splat.py" "$PLY" "$SPLAT_OUT"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done! $(date '+%Y-%m-%d %H:%M:%S')"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  $SPLAT_OUT"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Drag onto https://antimatter15.com/splat/ to view"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
