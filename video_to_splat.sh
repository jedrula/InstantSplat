#!/bin/bash
# video_to_splat.sh — full pipeline: video → frames → 3DGS → .splat
#
# Usage:
#   bash video_to_splat.sh <video> --n-frames N --duration S --iters N [options]
#
# Required:
#   --n-frames N     number of frames to extract
#   --duration S     seconds of video to sample (controls scene size)
#   --iters N        3DGS training iterations
#
# Optional:
#   --scene NAME     scene name / output folder (default: video filename stem)
#   --start T        start time in video, e.g. 00:00:05 (default: 0)
#
# Example (known-good for climbing wall scenes):
#   bash video_to_splat.sh /path/to/fpinka.mp4 --n-frames 3 --duration 6 --iters 1000

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/communications/miniconda3/envs/instantsplat/bin/python

# ── Arg parsing ──────────────────────────────────────────────────────────────
N_FRAMES=""
DURATION=""
ITERS=""
START=0
SCENE=""
VIDEO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)     SCENE="$2";    shift 2 ;;
        --n-frames)  N_FRAMES="$2"; shift 2 ;;
        --duration)  DURATION="$2"; shift 2 ;;
        --start)     START="$2";    shift 2 ;;
        --iters)     ITERS="$2";    shift 2 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           VIDEO="$1";    shift ;;
    esac
done

USAGE="Usage: bash video_to_splat.sh <video> --n-frames N --duration S --iters N [--scene NAME] [--start T]"

[[ -z "$VIDEO" ]]    && { echo "$USAGE"; exit 1; }
[[ -z "$N_FRAMES" ]] && { echo "Error: --n-frames is required"; echo "$USAGE"; exit 1; }
[[ -z "$DURATION" ]] && { echo "Error: --duration is required"; echo "$USAGE"; exit 1; }
[[ -z "$ITERS" ]]    && { echo "Error: --iters is required";    echo "$USAGE"; exit 1; }

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
# Encode key params in filename so artifacts are self-describing
SPLAT_OUT="$MODEL_DIR/${SCENE}_f${N_FRAMES}_d${DURATION}s_i${ITERS}.splat"

# ── Step 1: extract frames ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  video_to_splat: $SCENE"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "[1/3] Extracting $N_FRAMES frames over ${DURATION}s (start=$START)..."

rm -rf "$IMAGE_DIR"
mkdir -p "$IMAGE_DIR"

# Sample N_FRAMES evenly across DURATION seconds — controls scene size independently of frame count
FPS=$(awk "BEGIN {printf \"%.4f\", $N_FRAMES / $DURATION}")

ffmpeg -y -loglevel error \
    -ss "$START" \
    -t "$DURATION" \
    -i "$VIDEO" \
    -vf "fps=$FPS" \
    -frames:v "$N_FRAMES" \
    "$IMAGE_DIR/frame_%04d.png"

ACTUAL_FRAMES=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | wc -l)
echo "    → $ACTUAL_FRAMES frames (PNG) → $IMAGE_DIR"

PIPELINE_START=$(date +%s)

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

# ── Save run metadata ─────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
ELAPSED_MIN=$(awk "BEGIN {printf \"%.1f\", $ELAPSED / 60}")

PARAMS_FILE="$MODEL_DIR/${SCENE}_f${N_FRAMES}_d${DURATION}s_i${ITERS}.params.json"
cat > "$PARAMS_FILE" <<EOF
{
  "scene":            "$SCENE",
  "video":            "$VIDEO",
  "start":            "$START",
  "duration":         $DURATION,
  "n_frames":         $N_FRAMES,
  "iters":            $ITERS,
  "actual_frames":    $ACTUAL_FRAMES,
  "splat":            "$SPLAT_OUT",
  "render_seconds":   $ELAPSED,
  "render_minutes":   $ELAPSED_MIN,
  "timestamp":        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done! $(date '+%Y-%m-%d %H:%M:%S')  (${ELAPSED_MIN} min)"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  $SPLAT_OUT"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Drag onto https://antimatter15.com/splat/ to view"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
