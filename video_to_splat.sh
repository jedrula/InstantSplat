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

set -eo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
# Override INSTANTSPLAT_PYTHON env var for non-default conda locations
PYTHON="${INSTANTSPLAT_PYTHON:-${HOME}/miniconda3/envs/instantsplat/bin/python}"

# ── Arg parsing ──────────────────────────────────────────────────────────────
ITERS=""
SCENE=""
EARLY_STOP=1
EVENTS_FILE=""
MAX_INIT_POINTS=""
SMART_FRAMES=0
SMART_FPS=5.0
SPARSE_PAIRS=0
SPARSE_GA=0
FRAMES_ONLY=0
IMAGE_SIZE=256
# Per-video lists (comma-separated, one entry per video)
FPS_LIST=""
NFRAMES_LIST=""
START_LIST=""
DURATION_LIST=""
VIDEOS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)        SCENE="$2";        shift 2 ;;
        --iters)        ITERS="$2";        shift 2 ;;
        --events-file)  EVENTS_FILE="$2";  shift 2 ;;
        --early-stop)   EARLY_STOP=1;      shift ;;
        --no-early-stop) EARLY_STOP=0;     shift ;;
        --max-init-points) MAX_INIT_POINTS="$2"; shift 2 ;;
        --smart-frames) SMART_FRAMES=1;    shift ;;
        --smart-fps)    SMART_FPS="$2";    shift 2 ;;
        --frames-only)  FRAMES_ONLY=1;     shift ;;
        --sparse-pairs) SPARSE_PAIRS=1;    shift ;;
        --sparse-ga)    SPARSE_GA=1;       shift ;;
        --image-size)   IMAGE_SIZE="$2";   shift 2 ;;
        --fps-list)     FPS_LIST="$2";     shift 2 ;;
        --nframes-list) NFRAMES_LIST="$2"; shift 2 ;;
        --start-list)   START_LIST="$2";   shift 2 ;;
        --duration-list) DURATION_LIST="$2"; shift 2 ;;
        -*)             echo "Unknown option: $1"; exit 1 ;;
        *)              VIDEOS+=("$1"); shift ;;
    esac
done

# ── Event emitter ────────────────────────────────────────────────────────────
# Usage: emit_event '{"event": "...", ...extra keys...}'
# Automatically injects "t" (seconds since PIPELINE_START) when available.
emit_event() {
    [[ -z "$EVENTS_FILE" ]] && return 0
    local payload="$1"
    local ts=0
    if [[ -n "${PIPELINE_START:-}" ]]; then
        ts=$(( $(date +%s) - PIPELINE_START ))
    fi
    # Inject t field: strip trailing } and append ,"t":N}
    echo "${payload%\}},\"t\":$ts}" >> "$EVENTS_FILE"
}

USAGE="Usage:
  bash video_to_splat.sh <video> [video2 ...] --scene NAME --iters N
    --fps-list F1,F2,...   OR  --nframes-list N1,N2,...
    --start-list S1,S2,...  --duration-list D1,D2,..."

[[ ${#VIDEOS[@]} -eq 0 ]] && { echo "$USAGE"; exit 1; }
[[ -z "$SCENE" ]]          && { echo "Error: --scene is required"; echo "$USAGE"; exit 1; }
[[ -z "$ITERS" && "$FRAMES_ONLY" != "1" ]] && { echo "Error: --iters is required"; echo "$USAGE"; exit 1; }
ITERS="${ITERS:-0}"

if [[ -n "$FPS_LIST" && -n "$NFRAMES_LIST" ]]; then
    echo "Error: --fps-list and --nframes-list are mutually exclusive"; exit 1
fi
if [[ -z "$FPS_LIST" && -z "$NFRAMES_LIST" ]]; then
    echo "Error: provide either --fps-list or --nframes-list"; echo "$USAGE"; exit 1
fi

# Split comma-separated lists into arrays
IFS=',' read -ra _FPS_ARR      <<< "${FPS_LIST:-}"
IFS=',' read -ra _NFRAMES_ARR  <<< "${NFRAMES_LIST:-}"
IFS=',' read -ra _START_ARR    <<< "${START_LIST:-}"
IFS=',' read -ra _DURATION_ARR <<< "${DURATION_LIST:-}"

for VIDEO in "${VIDEOS[@]}"; do
    [[ ! -f "$VIDEO" ]] && { echo "Error: video not found: $VIDEO"; exit 1; }
done

SCENE_DIR="$REPO/assets/examples/$SCENE"
IMAGE_DIR="$SCENE_DIR/images"
MODEL_DIR="$REPO/output_infer/$SCENE"
mkdir -p "$MODEL_DIR"

# ── Step 1: extract frames from all videos into one dir ───────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  video_to_splat: $SCENE  (${#VIDEOS[@]} video(s))"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Watch logs:"
echo "║    tail -f $MODEL_DIR/01_init_geo.log"
echo "║    tail -f $MODEL_DIR/02_train.log"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

rm -rf "$IMAGE_DIR"
mkdir -p "$IMAGE_DIR"

FRAME_OFFSET=1
VIDEO_IDX=0
for VIDEO in "${VIDEOS[@]}"; do
    # Per-video start time (default 0)
    VSTART="${_START_ARR[$VIDEO_IDX]:-0}"
    [[ -z "$VSTART" ]] && VSTART=0

    # Per-video duration: use provided value, else probe full video minus start
    VDURATION="${_DURATION_ARR[$VIDEO_IDX]:-}"
    if [[ -z "$VDURATION" ]]; then
        FULL_DUR=$(ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "$VIDEO" | awk '{printf "%.3f", $1}')
        VDURATION=$(awk "BEGIN {printf \"%.3f\", $FULL_DUR - $VSTART}")
    fi

    # Per-video fps / n-frames
    if [[ -n "$FPS_LIST" ]]; then
        VFPS="${_FPS_ARR[$VIDEO_IDX]:-0.5}"
        VN_FRAMES=$(awk "BEGIN {printf \"%d\", int($VFPS * $VDURATION + 0.5)}")
    else
        VN_FRAMES="${_NFRAMES_ARR[$VIDEO_IDX]:-3}"
        VFPS=$(awk "BEGIN {printf \"%.4f\", $VN_FRAMES / $VDURATION}")
    fi

    if [[ "$SMART_FRAMES" == "1" ]]; then
        echo "[1/3] $(basename "$VIDEO"): smart-select $VN_FRAMES frames from ${VDURATION}s @ start=${VSTART}s (oversample ${SMART_FPS}fps, offset=$FRAME_OFFSET)..."
        TMP_SEL=$(mktemp -d)
        "$PYTHON" "$REPO/select_frames.py" "$VIDEO" \
            --fps "$SMART_FPS" \
            --target "$VN_FRAMES" \
            --start "$VSTART" \
            --duration "$VDURATION" \
            --out "$TMP_SEL"
        IN_SEL=$(ls "$TMP_SEL/selected/frame_"*.png 2>/dev/null | wc -l)
        echo ">> [frames] select_frames wrote $IN_SEL file(s) to $TMP_SEL/selected/"
        echo ">> [frames] exact contents: $(ls -1 "$TMP_SEL/selected/" 2>/dev/null | tr '\n' ' ')"
        VID_STEM=$(basename "$VIDEO" | sed 's/\.[^.]*$//')
        DEST_IDX=$FRAME_OFFSET
        for f in $(ls "$TMP_SEL/selected/frame_"*.png 2>/dev/null | sort); do
            # Compute timestamp: frame_NNNN → index NNNN-1 → time = (NNNN-1)/SMART_FPS + VSTART
            FNUM=$(basename "$f" | sed 's/frame_0*\([0-9]*\)\.png/\1/')
            TSEC=$(awk "BEGIN {printf \"%07.2f\", ($FNUM - 1) / $SMART_FPS + $VSTART}")
            DESTNAME="${SCENE}_${VID_STEM}_t${TSEC}s.png"
            cp "$f" "$IMAGE_DIR/$DESTNAME"
            DEST_IDX=$(( DEST_IDX + 1 ))
        done
        COPIED=$(( DEST_IDX - FRAME_OFFSET ))
        echo ">> [frames] copied $COPIED frame(s) into $IMAGE_DIR (indices $FRAME_OFFSET..$(( DEST_IDX - 1 )))"
        rm -rf "$TMP_SEL"
    else
        echo "[1/3] $(basename "$VIDEO"): $VN_FRAMES frames over ${VDURATION}s at ${VFPS}fps (start=${VSTART}s, offset=$FRAME_OFFSET)..."
        ffmpeg -y -loglevel error \
            -ss "$VSTART" \
            -t "$VDURATION" \
            -i "$VIDEO" \
            -vf "fps=$VFPS" \
            -frames:v "$VN_FRAMES" \
            -start_number "$FRAME_OFFSET" \
            "$IMAGE_DIR/frame_%04d.png"
    fi

    EXTRACTED=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | wc -l)
    FRAME_OFFSET=$(( EXTRACTED + 1 ))
    VIDEO_IDX=$(( VIDEO_IDX + 1 ))
done

TOTAL_FRAMES=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | wc -l)
echo "    → $TOTAL_FRAMES total frames (PNG) → $IMAGE_DIR"
echo ">> [frames] all files in IMAGE_DIR: $(ls "$IMAGE_DIR"/*.png 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

if [[ "$FRAMES_ONLY" == "1" ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  --frames-only: stopping after frame extraction"
    echo "║  $TOTAL_FRAMES frames in $IMAGE_DIR"
    echo "╚══════════════════════════════════════════════════════╝"
    exit 0
fi

PIPELINE_START=$(date +%s)

emit_event "{\"event\":\"frames_extracted\",\"total_frames\":$TOTAL_FRAMES}"

# ── Step 2: geometry init + 3DGS training ────────────────────────────────────
echo ""
echo "[2/3] MASt3R init + 3DGS training ($ITERS iterations, $TOTAL_FRAMES frames)..."
mkdir -p "$MODEL_DIR"
cd "$REPO"

MAST3R_START=$(date +%s)
# Clear any cache left by a previous crashed run; stale SparseGA cache causes
# device-side assert when tensor shapes no longer match the new image set.
rm -rf "$MODEL_DIR/sparse_ga_cache"
INIT_GEO_ARGS=""
[[ -n "$MAX_INIT_POINTS" ]] && INIT_GEO_ARGS="--max_init_points $MAX_INIT_POINTS"
[[ "$SPARSE_PAIRS" == "1" ]] && INIT_GEO_ARGS="$INIT_GEO_ARGS --sparse_pairs"
[[ "$SPARSE_GA"    == "1" ]] && INIT_GEO_ARGS="$INIT_GEO_ARGS --sparse_ga"
# image_size controls MASt3R pose estimation resolution; train.py always uses full-res originals.
# 256 = 40+ frames on 8GB GPU; 512 = best init quality but OOMs above ~14 frames.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -W ignore ./init_geo.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    --n_views "$TOTAL_FRAMES" \
    --image_size "$IMAGE_SIZE" \
    --focal_avg \
    --co_vis_dsp \
    --conf_aware_ranking \
    --infer_video \
    $INIT_GEO_ARGS \
    2>&1 | tee "$MODEL_DIR/01_init_geo.log"
MAST3R_END=$(date +%s)
MAST3R_ELAPSED=$(( MAST3R_END - MAST3R_START ))
MAST3R_MIN=$(awk "BEGIN {printf \"%.1f\", $MAST3R_ELAPSED / 60}")
echo "    → MASt3R step: ${MAST3R_ELAPSED}s (${MAST3R_MIN} min)"

emit_event "{\"event\":\"mast3r_done\",\"elapsed_s\":$MAST3R_ELAPSED}"

if (( MAST3R_ELAPSED > 120 )); then
    echo "    ⚠  Slow MASt3R (>${MAST3R_ELAPSED}s) — compiling the RoPE2D CUDA kernel would save ~20-40% here. See InstantSplat/TODO.md."
else
    echo "    ✓  MASt3R fast enough — RoPE2D CUDA kernel not worth optimizing yet."
fi

EARLY_STOP_ARGS=""
if [[ "$EARLY_STOP" == "1" ]]; then
    # Smart-frames scenes cover more of the wall — far cameras need more iters
    # to converge, so double the patience to avoid stopping too early.
    PATIENCE=200
    [[ "$SMART_FRAMES" == "1" ]] && PATIENCE=400
    EARLY_STOP_ARGS="--early_stop_patience $PATIENCE --early_stop_delta 1e-5"
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" ./train.py \
    -s "$SCENE_DIR" \
    -m "$MODEL_DIR" \
    -r 1 \
    --n_views "$TOTAL_FRAMES" \
    --iterations "$ITERS" \
    --pp_optimizer \
    --optim_pose \
    $EARLY_STOP_ARGS \
    2>&1 | tee "$MODEL_DIR/02_train.log"

echo "    → done"

# ── Emit train_done event (grep tee'd log for early-stop info) ───────────────
TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - MAST3R_END ))
EARLY_STOPPED_FLAG="false"
STOPPED_AT_ITER="null"
if grep -q 'Early stopping' "$MODEL_DIR/02_train.log" 2>/dev/null; then
    EARLY_STOPPED_FLAG="true"
    STOPPED_AT_ITER=$(grep -oP '\[ITER \K\d+(?=\] Early stopping)' "$MODEL_DIR/02_train.log" | tail -1)
    STOPPED_AT_ITER=${STOPPED_AT_ITER:-null}
fi
emit_event "{\"event\":\"train_done\",\"elapsed_s\":$TRAIN_ELAPSED,\"early_stopped\":$EARLY_STOPPED_FLAG,\"stopped_at_iter\":$STOPPED_AT_ITER}"

# Resolve actual iteration after training (early stop may save before ITERS)
ACTUAL_ITER=$(ls -d "$MODEL_DIR/point_cloud/iteration_"* 2>/dev/null \
    | grep -oP 'iteration_\K\d+' | sort -n | tail -1)
ACTUAL_ITER=${ACTUAL_ITER:-$ITERS}
PLY="$MODEL_DIR/point_cloud/iteration_${ACTUAL_ITER}/point_cloud.ply"
SPLAT_OUT="$MODEL_DIR/${SCENE}_f${TOTAL_FRAMES}_i${ACTUAL_ITER}.splat"

# ── Step 3: PLY → .splat ─────────────────────────────────────────────────────
echo ""
echo "[3/3] Converting to .splat..."
PLY2SPLAT_START=$(date +%s)
"$PYTHON" "$REPO/ply2splat.py" "$PLY" "$SPLAT_OUT"
PLY2SPLAT_ELAPSED=$(( $(date +%s) - PLY2SPLAT_START ))
SPLAT_SIZE=$(stat -c%s "$SPLAT_OUT" 2>/dev/null || echo 0)
emit_event "{\"event\":\"splat_ready\",\"filename\":\"$(basename $SPLAT_OUT)\",\"size_bytes\":$SPLAT_SIZE,\"ply2splat_s\":$PLY2SPLAT_ELAPSED}"

# ── Save run metadata ─────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
ELAPSED_MIN=$(awk "BEGIN {printf \"%.1f\", $ELAPSED / 60}")

PARAMS_FILE="$MODEL_DIR/${SCENE}_f${TOTAL_FRAMES}_i${ACTUAL_ITER}.params.json"
cat > "$PARAMS_FILE" <<EOF
{
  "scene":            "$SCENE",
  "videos":           [$(printf '"%s",' "${VIDEOS[@]}" | sed 's/,$//')]  ,
  "start_list":       "$START_LIST",
  "duration_list":    "$DURATION_LIST",
  "fps_list":         "$FPS_LIST",
  "nframes_list":     "$NFRAMES_LIST",
  "total_frames":     $TOTAL_FRAMES,
  "iters":            $ACTUAL_ITER,
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
