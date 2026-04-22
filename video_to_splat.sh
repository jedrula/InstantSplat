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

# Default scene name = video filename without extension
if [[ -z "$SCENE" ]]; then
    SCENE="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
fi

SCENE_DIR="$REPO/assets/examples/$SCENE"
IMAGE_DIR="$SCENE_DIR/images"
MODEL_DIR="$REPO/output_infer/$SCENE"
SPLAT_OUT="$MODEL_DIR/$SCENE.splat"

# ── Step 0: extract frames ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  video_to_splat: $SCENE"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "[0/4] Extracting $N_FRAMES frames from $VIDEO (start=$START)..."

rm -rf "$IMAGE_DIR"
mkdir -p "$IMAGE_DIR"

# Get video duration to compute frame interval
DURATION=$(ffprobe -v error -ss "$START" -show_entries format=duration \
    -of csv=p=0 "$VIDEO" 2>/dev/null | awk '{printf "%.2f", $1}')
INTERVAL=$(echo "$DURATION $N_FRAMES" | awk '{printf "%.4f", $1/$2}')

ffmpeg -y -loglevel error \
    -ss "$START" \
    -i "$VIDEO" \
    -vf "fps=1/$INTERVAL" \
    -frames:v "$N_FRAMES" \
    -q:v 2 \
    "$IMAGE_DIR/frame_%04d.jpg"

ACTUAL=$(ls "$IMAGE_DIR"/*.jpg 2>/dev/null | wc -l)
echo "    → $ACTUAL frames saved to $IMAGE_DIR"

# ── Step 1: geometry init (MASt3R) ───────────────────────────────────────────
echo ""
echo "[1/4] MASt3R geometry init ($N_FRAMES views)..."
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
echo "    → done (log: $MODEL_DIR/01_init_geo.log)"

# ── Step 2: 3DGS training ────────────────────────────────────────────────────
echo ""
echo "[2/4] Training 3DGS ($ITERS iterations)..."

CUDA_VISIBLE_DEVICES=0 "$PYTHON" ./train.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    -r 1 \
    --n_views "$N_FRAMES" \
    --iterations "$ITERS" \
    --pp_optimizer \
    --optim_pose \
    2>&1 | tee "$MODEL_DIR/02_train.log"
echo "    → done (log: $MODEL_DIR/02_train.log)"

# ── Step 3: render ────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Rendering..."

CUDA_VISIBLE_DEVICES=0 "$PYTHON" ./render.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    -r 1 \
    --n_views "$N_FRAMES" \
    --iterations "$ITERS" \
    --infer_video \
    2>&1 | tee "$MODEL_DIR/03_render.log"

# Assemble PNG renders into video
RENDERS="$MODEL_DIR/interp/ours_${ITERS}/renders"
if ls "$RENDERS"/*.png &>/dev/null; then
    ffmpeg -y -loglevel error \
        -framerate 24 \
        -pattern_type glob -i "$RENDERS/*.png" \
        -c:v libx264 -pix_fmt yuv420p \
        "$MODEL_DIR/result.mp4"
    echo "    → video: $MODEL_DIR/result.mp4"
fi
echo "    → done (log: $MODEL_DIR/03_render.log)"

# ── Step 4: PLY → .splat ─────────────────────────────────────────────────────
echo ""
echo "[4/4] Converting to .splat..."

PLY="$MODEL_DIR/point_cloud/iteration_${ITERS}/point_cloud.ply"
"$PYTHON" "$REPO/ply2splat.py" "$PLY" "$SPLAT_OUT"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done!"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Splat file:  $SPLAT_OUT"
echo "║  Video:       $MODEL_DIR/result.mp4"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  View in browser — drag the splat file onto:"
echo "║    https://antimatter15.com/splat/"
echo "║  (or for annotations: https://playcanvas.com/supersplat/editor)"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
