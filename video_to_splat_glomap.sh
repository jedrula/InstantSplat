#!/bin/bash
# video_to_splat_glomap.sh — GLOMAP+gsplat pipeline: video(s) → frames → COLMAP feats → GLOMAP → 3DGS → .splat
#
# Single video:
#   bash video_to_splat_glomap.sh <video> --scene NAME --n-frames N --duration S --iters N
#   bash video_to_splat_glomap.sh <video> --scene NAME --fps F --iters N [--duration S]
#
# Multiple videos (frames merged into one scene):
#   bash video_to_splat_glomap.sh v1.mp4 v2.mp4 --scene NAME --n-frames N --duration S --iters N
#
# Required:
#   --scene NAME     scene name / output folder
#   --iters N        gsplat training iterations
#   mode A: --n-frames N  AND  --duration S
#   mode B: --fps F   (--duration optional)
#
# Optional:
#   --start T          start time in video, e.g. 00:00:05 (default: 0)
#   --camera-model M   COLMAP camera model (default: SIMPLE_RADIAL; use PINHOLE for undistorted footage)
#   --events-file F    NDJSON events file
#
# Examples:
#   bash video_to_splat_glomap.sh v.mp4 --scene wall --n-frames 20 --duration 10 --iters 5000
#   bash video_to_splat_glomap.sh v1.mp4 v2.mp4 --scene wall --n-frames 10 --duration 4 --iters 5000

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${INSTANTSPLAT_PYTHON:-${HOME}/miniconda3/envs/instantsplat/bin/python}"
GLOMAP_BIN="${GLOMAP_BIN:-${HOME}/miniconda3/envs/instantsplat/bin/glomap}"

# ── Arg parsing ──────────────────────────────────────────────────────────────
N_FRAMES=""
DURATION=""
FPS_ARG=""
ITERS=""
START=0
SCENE=""
EVENTS_FILE=""
CAMERA_MODEL="SIMPLE_RADIAL"
VIDEOS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)         SCENE="$2";         shift 2 ;;
        --n-frames)      N_FRAMES="$2";      shift 2 ;;
        --duration)      DURATION="$2";      shift 2 ;;
        --fps)           FPS_ARG="$2";       shift 2 ;;
        --start)         START="$2";         shift 2 ;;
        --iters)         ITERS="$2";         shift 2 ;;
        --events-file)   EVENTS_FILE="$2";   shift 2 ;;
        --camera-model)  CAMERA_MODEL="$2";  shift 2 ;;
        -*)              echo "Unknown option: $1"; exit 1 ;;
        *)               VIDEOS+=("$1"); shift ;;
    esac
done

# ── Event emitter ────────────────────────────────────────────────────────────
emit_event() {
    [[ -z "$EVENTS_FILE" ]] && return 0
    local payload="$1"
    local ts=0
    [[ -n "${PIPELINE_START:-}" ]] && ts=$(( $(date +%s) - PIPELINE_START ))
    echo "${payload%\}},\"t\":$ts}" >> "$EVENTS_FILE"
}

USAGE="Usage: bash video_to_splat_glomap.sh <video> [video2 ...] --scene NAME --iters N (--n-frames N --duration S | --fps F)"
[[ ${#VIDEOS[@]} -eq 0 ]] && { echo "$USAGE"; exit 1; }
[[ -z "$SCENE" ]]          && { echo "Error: --scene required"; exit 1; }
[[ -z "$ITERS" ]]          && { echo "Error: --iters required"; exit 1; }
[[ -n "$FPS_ARG" && -n "$N_FRAMES" ]] && { echo "Error: --fps and --n-frames are mutually exclusive"; exit 1; }
[[ -z "$FPS_ARG" && ( -z "$N_FRAMES" || -z "$DURATION" ) ]] && { echo "Error: provide --fps F or both --n-frames N --duration S"; exit 1; }
for V in "${VIDEOS[@]}"; do [[ ! -f "$V" ]] && { echo "Error: not found: $V"; exit 1; }; done

SCENE_DIR="$REPO/assets/examples/$SCENE"
IMAGE_DIR="$SCENE_DIR/images"
DB_PATH="$SCENE_DIR/database.db"
# GLOMAP writes to $SPARSE_PARENT and creates sub-dir 0/ → gsplat Parser finds sparse/0/ automatically
SPARSE_PARENT="$SCENE_DIR/sparse"
MODEL_DIR="$REPO/output_glomap/$SCENE"
mkdir -p "$MODEL_DIR"

# ── Step 1: Extract frames ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  video_to_splat_glomap: $SCENE  (${#VIDEOS[@]} video(s))"
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

    echo "[1/4] $(basename "$VIDEO"): $VN_FRAMES frames over ${VDURATION}s at ${VFPS}fps (start=$START, offset=$FRAME_OFFSET)..."
    ffmpeg -y -loglevel error \
        -ss "$START" -t "$VDURATION" -i "$VIDEO" \
        -vf "fps=$VFPS" -frames:v "$VN_FRAMES" \
        -start_number "$FRAME_OFFSET" \
        "$IMAGE_DIR/frame_%04d.png"
    EXTRACTED=$(ls "$IMAGE_DIR"/frame_*.png 2>/dev/null | wc -l)
    FRAME_OFFSET=$(( EXTRACTED + 1 ))
done

TOTAL_FRAMES=$(ls "$IMAGE_DIR"/frame_*.png 2>/dev/null | wc -l)
echo "    → $TOTAL_FRAMES total frames → $IMAGE_DIR"

PIPELINE_START=$(date +%s)
emit_event "{\"event\":\"frames_extracted\",\"total_frames\":$TOTAL_FRAMES}"

# ── Step 2: COLMAP feature extraction + matching ──────────────────────────────
# Matcher choice:
#   ≤ 50 frames → exhaustive_matcher: tries all N*(N-1)/2 pairs (190 for 20 frames)
#                 Best quality; finds matches across all overlapping frames even
#                 when sequential matching misses them (e.g. repetitive textures).
#   > 50 frames → sequential_matcher: efficient for long video; overlap=10 bridges
#                 multi-video segments. Falls short on repetitive textures.
echo ""
if (( TOTAL_FRAMES <= 50 )); then
    MATCHER_DESC="exhaustive"
else
    MATCHER_DESC="sequential (overlap=10)"
fi
echo "[2/4] COLMAP features + ${MATCHER_DESC} matching ($TOTAL_FRAMES images, camera=$CAMERA_MODEL)..."

rm -f "$DB_PATH"
COLMAP_START=$(date +%s)

colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.camera_model "$CAMERA_MODEL" \
    --ImageReader.single_camera 1 \
    2>&1 | tee "$MODEL_DIR/01a_colmap_features.log"

if (( TOTAL_FRAMES <= 50 )); then
    colmap exhaustive_matcher \
        --database_path "$DB_PATH" \
        2>&1 | tee "$MODEL_DIR/01b_colmap_match.log"
else
    colmap sequential_matcher \
        --database_path "$DB_PATH" \
        --SequentialMatching.overlap 10 \
        --SequentialMatching.loop_detection 0 \
        2>&1 | tee "$MODEL_DIR/01b_colmap_match.log"
fi

COLMAP_ELAPSED=$(( $(date +%s) - COLMAP_START ))
echo "    → COLMAP features+matching: ${COLMAP_ELAPSED}s"
emit_event "{\"event\":\"colmap_done\",\"elapsed_s\":$COLMAP_ELAPSED}"

# ── Step 3: SfM — COLMAP incremental mapper ──────────────────────────────────
# COLMAP incremental instead of GLOMAP global SfM:
#   GLOMAP rotation-averaging fails silently on near-planar scenes (e.g. climbing
#   walls). One bad frame gets placed 12000 units away, poisoning all training.
#   COLMAP incremental is frame-by-frame, detects degeneracies, more robust for
#   <~200 images. Re-enable GLOMAP here if you move to large photo collections.
echo ""
echo "[3/4] COLMAP incremental SfM..."

rm -rf "$SPARSE_PARENT"
mkdir -p "$SPARSE_PARENT"

SFM_START=$(date +%s)
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_PARENT" \
    2>&1 | tee "$MODEL_DIR/02_sfm.log"
SFM_ELAPSED=$(( $(date +%s) - SFM_START ))
echo "    → COLMAP mapper: ${SFM_ELAPSED}s"
emit_event "{\"event\":\"sfm_done\",\"elapsed_s\":$SFM_ELAPSED}"

if [[ ! -d "$SPARSE_PARENT/0" ]]; then
    echo "Error: COLMAP produced no reconstruction in $SPARSE_PARENT/0"
    echo "Check $MODEL_DIR/02_sfm.log for details."
    exit 1
fi

N_COMPONENTS=$(ls -d "$SPARSE_PARENT"/[0-9]* 2>/dev/null | wc -l)
(( N_COMPONENTS > 1 )) && echo "    ⚠  $N_COMPONENTS disconnected components — using component 0 (largest)"

# ── Diagnose and filter outlier cameras ──────────────────────────────────────
# A single misregistered camera (position >3σ from centroid) can scatter all
# Gaussians. We detect outliers, report them, and move their image files out so
# gsplat only trains on the clean subset.
SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
"$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1

echo "    → Sparse reconstruction: $SPARSE_PARENT/0/"

# ── Step 4: gsplat training ───────────────────────────────────────────────────
echo ""
echo "[4/4] gsplat 3DGS training ($ITERS iterations)..."

TRAIN_START=$(date +%s)
PYTHONPATH="$REPO:$REPO/gsplat_examples" CUDA_VISIBLE_DEVICES=0 \
"$PYTHON" "$REPO/simple_trainer.py" default \
    --data_dir "$SCENE_DIR" \
    --result_dir "$MODEL_DIR" \
    --max_steps "$ITERS" \
    --data_factor 1 \
    --disable_viewer \
    --save_ply \
    2>&1 | tee "$MODEL_DIR/03_train.log"
TRAIN_ELAPSED=$(( $(date +%s) - TRAIN_START ))
echo "    → gsplat training: ${TRAIN_ELAPSED}s"
emit_event "{\"event\":\"train_done\",\"elapsed_s\":$TRAIN_ELAPSED}"

# ── Step 5: PLY → .splat ─────────────────────────────────────────────────────
LAST_STEP=$(( ITERS - 1 ))
PLY="$MODEL_DIR/ply/point_cloud_${LAST_STEP}.ply"
[[ ! -f "$PLY" ]] && PLY=$(ls -t "$MODEL_DIR/ply/"*.ply 2>/dev/null | head -1)
if [[ -z "$PLY" || ! -f "$PLY" ]]; then
    echo "Error: no PLY found in $MODEL_DIR/ply/"; exit 1
fi

SPLAT_OUT="$MODEL_DIR/${SCENE}_f${TOTAL_FRAMES}_i${ITERS}.splat"
echo ""
echo "[5/5] Converting $(basename "$PLY") → .splat..."
PLY2SPLAT_START=$(date +%s)
"$PYTHON" "$REPO/ply2splat.py" "$PLY" "$SPLAT_OUT"
PLY2SPLAT_ELAPSED=$(( $(date +%s) - PLY2SPLAT_START ))
SPLAT_SIZE=$(stat -c%s "$SPLAT_OUT" 2>/dev/null || echo 0)
emit_event "{\"event\":\"splat_ready\",\"filename\":\"$(basename $SPLAT_OUT)\",\"size_bytes\":$SPLAT_SIZE,\"ply2splat_s\":$PLY2SPLAT_ELAPSED}"

# ── Save run metadata ─────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
ELAPSED_MIN=$(awk "BEGIN {printf \"%.1f\", $ELAPSED / 60}")
PARAMS_FILE="$MODEL_DIR/${SCENE}_f${TOTAL_FRAMES}_i${ITERS}.params.json"
cat > "$PARAMS_FILE" <<EOF
{
  "scene":            "$SCENE",
  "pipeline":         "glomap+gsplat",
  "videos":           [$(printf '"%s",' "${VIDEOS[@]}" | sed 's/,$//')]  ,
  "start":            "$START",
  "duration":         ${VDURATION:-null},
  "fps":              ${VFPS:-null},
  "n_frames_per_video": ${VN_FRAMES:-null},
  "total_frames":     $TOTAL_FRAMES,
  "camera_model":     "$CAMERA_MODEL",
  "iters":            $ITERS,
  "splat":            "$SPLAT_OUT",
  "colmap_seconds":   $COLMAP_ELAPSED,
  "glomap_seconds":   $GLOMAP_ELAPSED,
  "train_seconds":    $TRAIN_ELAPSED,
  "render_seconds":   $ELAPSED,
  "render_minutes":   $ELAPSED_MIN,
  "timestamp":        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done! $(date '+%Y-%m-%d %H:%M:%S')  (${ELAPSED_MIN} min)"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  $SPLAT_OUT"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Drag onto https://antimatter15.com/splat/ to view"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
