#!/bin/bash
# video_to_splat.sh — full pipeline: video(s) → frames → 3DGS → .splat
#
# Single video:
#   bash video_to_splat.sh <video> --scene NAME --n-frames N --duration S --iters N
#   bash video_to_splat.sh <video> --scene NAME --fps F --iters N [--duration S]
#
# Multiple videos (frames merged into one scene):
#   bash video_to_splat.sh v1.mp4 v2.mp4 --scene NAME --n-frames N --duration S --iters N
#   (--n-frames / --duration apply per video; total frames = n_videos × n_frames)
#
# Required:
#   --scene NAME     scene name / output folder (always required)
#   --iters N        3DGS training iterations
#   mode A: --n-frames N  AND  --duration S
#   mode B: --fps F   (--duration optional; omit to use full video length)
#
# Optional:
#   --start T        start time in video, e.g. 00:00:05 (default: 0)
#
# Examples:
#   bash video_to_splat.sh v.mp4 --scene wall --n-frames 3 --duration 6 --iters 1000
#   bash video_to_splat.sh v1.mp4 v2.mp4 --scene wall --n-frames 3 --duration 6 --iters 1000

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/communications/miniconda3/envs/instantsplat/bin/python

# ── Arg parsing ──────────────────────────────────────────────────────────────
N_FRAMES=""
DURATION=""
FPS_ARG=""
ITERS=""
START=0
SCENE=""
VIDEOS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)     SCENE="$2";    shift 2 ;;
        --n-frames)  N_FRAMES="$2"; shift 2 ;;
        --duration)  DURATION="$2"; shift 2 ;;
        --fps)       FPS_ARG="$2";  shift 2 ;;
        --start)     START="$2";    shift 2 ;;
        --iters)     ITERS="$2";    shift 2 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           VIDEOS+=("$1"); shift ;;
    esac
done

USAGE="Usage:
  bash video_to_splat.sh <video> [video2 ...] --scene NAME --iters N
    mode A: --n-frames N --duration S
    mode B: --fps F [--duration S]"

[[ ${#VIDEOS[@]} -eq 0 ]] && { echo "$USAGE"; exit 1; }
[[ -z "$SCENE" ]]          && { echo "Error: --scene is required"; echo "$USAGE"; exit 1; }
[[ -z "$ITERS" ]]          && { echo "Error: --iters is required"; echo "$USAGE"; exit 1; }

if [[ -n "$FPS_ARG" && -n "$N_FRAMES" ]]; then
    echo "Error: --fps and --n-frames are mutually exclusive"; exit 1
fi
if [[ -z "$FPS_ARG" && ( -z "$N_FRAMES" || -z "$DURATION" ) ]]; then
    echo "Error: provide either --fps F or both --n-frames N and --duration S"
    echo "$USAGE"; exit 1
fi

for VIDEO in "${VIDEOS[@]}"; do
    [[ ! -f "$VIDEO" ]] && { echo "Error: video not found: $VIDEO"; exit 1; }
done

SCENE_DIR="$REPO/assets/examples/$SCENE"
IMAGE_DIR="$SCENE_DIR/images"
MODEL_DIR="$REPO/output_infer/$SCENE"

# ── Step 1: extract frames from all videos into one dir ───────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  video_to_splat: $SCENE  (${#VIDEOS[@]} video(s))"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

rm -rf "$IMAGE_DIR"
mkdir -p "$IMAGE_DIR"

FRAME_OFFSET=1
for VIDEO in "${VIDEOS[@]}"; do
    if [[ -n "$FPS_ARG" ]]; then
        VDURATION="$DURATION"
        if [[ -z "$VDURATION" ]]; then
            VDURATION=$(ffprobe -v error -show_entries format=duration \
                -of default=noprint_wrappers=1:nokey=1 "$VIDEO" | awk '{printf "%.0f", $1}')
        fi
        VN_FRAMES=$(awk "BEGIN {printf \"%d\", int($FPS_ARG * $VDURATION + 0.5)}")
        VFPS=$FPS_ARG
    else
        VDURATION="$DURATION"
        VN_FRAMES="$N_FRAMES"
        VFPS=$(awk "BEGIN {printf \"%.4f\", $VN_FRAMES / $VDURATION}")
    fi

    echo "[1/3] $(basename "$VIDEO"): $VN_FRAMES frames over ${VDURATION}s at ${VFPS}fps (start=$START, offset=$FRAME_OFFSET)..."

    ffmpeg -y -loglevel error \
        -ss "$START" \
        -t "$VDURATION" \
        -i "$VIDEO" \
        -vf "fps=$VFPS" \
        -frames:v "$VN_FRAMES" \
        -start_number "$FRAME_OFFSET" \
        "$IMAGE_DIR/frame_%04d.png"

    EXTRACTED=$(ls "$IMAGE_DIR"/frame_*.png 2>/dev/null | wc -l)
    FRAME_OFFSET=$(( EXTRACTED + 1 ))
done

TOTAL_FRAMES=$(ls "$IMAGE_DIR"/frame_*.png 2>/dev/null | wc -l)
echo "    → $TOTAL_FRAMES total frames (PNG) → $IMAGE_DIR"

PLY="$MODEL_DIR/point_cloud/iteration_${ITERS}/point_cloud.ply"
SPLAT_OUT="$MODEL_DIR/${SCENE}_f${TOTAL_FRAMES}_i${ITERS}.splat"

PIPELINE_START=$(date +%s)

# ── Step 2: geometry init + 3DGS training ────────────────────────────────────
echo ""
echo "[2/3] MASt3R init + 3DGS training ($ITERS iterations, $TOTAL_FRAMES frames)..."
mkdir -p "$MODEL_DIR"
cd "$REPO"

MAST3R_START=$(date +%s)
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -W ignore ./init_geo.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    --n_views "$TOTAL_FRAMES" \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    --infer_video \
    2>&1 | tee "$MODEL_DIR/01_init_geo.log"
MAST3R_END=$(date +%s)
MAST3R_ELAPSED=$(( MAST3R_END - MAST3R_START ))
MAST3R_MIN=$(awk "BEGIN {printf \"%.1f\", $MAST3R_ELAPSED / 60}")
echo "    → MASt3R step: ${MAST3R_ELAPSED}s (${MAST3R_MIN} min)"
if (( MAST3R_ELAPSED > 120 )); then
    echo "    ⚠  Slow MASt3R (>${MAST3R_ELAPSED}s) — compiling the RoPE2D CUDA kernel would save ~20-40% here. See InstantSplat/TODO.md."
else
    echo "    ✓  MASt3R fast enough — RoPE2D CUDA kernel not worth optimizing yet."
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" ./train.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    -r 1 \
    --n_views "$TOTAL_FRAMES" \
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

PARAMS_FILE="$MODEL_DIR/${SCENE}_f${TOTAL_FRAMES}_i${ITERS}.params.json"
cat > "$PARAMS_FILE" <<EOF
{
  "scene":            "$SCENE",
  "videos":           [$(printf '"%s",' "${VIDEOS[@]}" | sed 's/,$//')]  ,
  "start":            "$START",
  "duration":         ${VDURATION:-null},
  "fps":              ${VFPS:-null},
  "n_frames_per_video": ${VN_FRAMES:-null},
  "total_frames":     $TOTAL_FRAMES,
  "iters":            $ITERS,
  "splat":            "$SPLAT_OUT",
  "mast3r_seconds":   $MAST3R_ELAPSED,
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
