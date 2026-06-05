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
# Skip frame extraction (images already in assets/examples/$SCENE/images/):
#   bash video_to_splat.sh --skip-extraction --scene NAME --iters N [options]
#
# Required:
#   --scene NAME     scene name / output folder (always required)
#   --iters N        3DGS training iterations
#   mode A: --n-frames N  AND  --duration S
#   mode B: --fps F   (--duration optional; omit to use full video length)
#   mode C: --skip-extraction  (no video files needed; fps/n-frames also not needed)
#
# Optional:
#   --start T        start time in video, e.g. 00:00:05 (default: 0)
#
# Examples:
#   bash video_to_splat.sh v.mp4 --scene wall --n-frames 3 --duration 6 --iters 1000
#   bash video_to_splat.sh v1.mp4 v2.mp4 --scene wall --n-frames 3 --duration 6 --iters 1000
#   bash video_to_splat.sh --skip-extraction --scene wall --iters 1000

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
SFM="mast3r"       # mast3r | fast3r | colmap
TRAINER="instantsplat"  # instantsplat | pgsr | splatfacto | gsplat
COLMAP_BA=0
COLMAP_MATCHER=""
NO_POINT_CAP=0
NO_DENSIFICATION=0
FRAMES_ONLY=0
SKIP_EXTRACTION=0
IMAGE_SIZE=256
MODEL_DIR_OVERRIDE=""
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
        --skip-extraction) SKIP_EXTRACTION=1; shift ;;
        --sparse-pairs) SPARSE_PAIRS=1;    shift ;;
        --sparse-ga)    SPARSE_GA=1;       shift ;;
        --sfm)              SFM="$2";            shift 2 ;;
        --trainer)          TRAINER="$2";        shift 2 ;;
        --engine)           # deprecated: map to --sfm + --trainer
            case "$2" in
                pgsr)   SFM="mast3r";  TRAINER="pgsr" ;;
                fast3r) SFM="fast3r";  TRAINER="instantsplat" ;;
                colmap) SFM="colmap";  TRAINER="instantsplat" ;;
                *)      SFM="mast3r";  TRAINER="instantsplat" ;;
            esac
            shift 2 ;;
        --colmap-ba)        COLMAP_BA=1;         shift ;;
        --colmap-matcher)   COLMAP_MATCHER="$2"; shift 2 ;;
        --no-point-cap)      NO_POINT_CAP=1;      shift ;;
        --no-densification)  NO_DENSIFICATION=1;  shift ;;
        --image-size)   IMAGE_SIZE="$2";   shift 2 ;;
        --model-dir)    MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
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

[[ "$SKIP_EXTRACTION" != "1" && ${#VIDEOS[@]} -eq 0 ]] && { echo "$USAGE"; exit 1; }
[[ -z "$SCENE" ]]          && { echo "Error: --scene is required"; echo "$USAGE"; exit 1; }
[[ -z "$ITERS" && "$FRAMES_ONLY" != "1" ]] && { echo "Error: --iters is required"; echo "$USAGE"; exit 1; }
ITERS="${ITERS:-0}"

if [[ "$SKIP_EXTRACTION" != "1" ]]; then
    if [[ -n "$FPS_LIST" && -n "$NFRAMES_LIST" ]]; then
        echo "Error: --fps-list and --nframes-list are mutually exclusive"; exit 1
    fi
    if [[ -z "$FPS_LIST" && -z "$NFRAMES_LIST" ]]; then
        echo "Error: provide either --fps-list or --nframes-list"; echo "$USAGE"; exit 1
    fi
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
# --model-dir lets callers (e.g. the server) isolate outputs per-job;
# manual runs default to the shared output_infer/$SCENE/ directory.
MODEL_DIR="${MODEL_DIR_OVERRIDE:-$REPO/output_infer/$SCENE}"
mkdir -p "$MODEL_DIR"

if [[ "$SKIP_EXTRACTION" == "1" ]]; then
    # ── Skip frame extraction: images already in IMAGE_DIR ───────────────────
    if [[ ! -d "$IMAGE_DIR" ]]; then
        echo "Error: --skip-extraction requires images already in $IMAGE_DIR"
        exit 1
    fi
    TOTAL_FRAMES=$(ls "$IMAGE_DIR" 2>/dev/null | grep -cE '\.(jpg|jpeg|png|webp)$' || true)
    if [[ "$TOTAL_FRAMES" -lt 2 ]]; then
        echo "Error: need at least 2 images in $IMAGE_DIR (found $TOTAL_FRAMES)"
        exit 1
    fi
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  video_to_splat (--skip-extraction): $SCENE"
    echo "║  Using $TOTAL_FRAMES pre-supplied image(s) from:"
    echo "║    $IMAGE_DIR"
    echo "╠══════════════════════════════════════════════════════╣"
    echo "║  Watch logs:"
    echo "║    tail -f $MODEL_DIR/01_init_geo.log"
    echo "║    tail -f $MODEL_DIR/02_train.log"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""
else
    # ── Step 1: extract frames from all videos into one dir ──────────────────
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
fi  # end of extraction block

PIPELINE_START=$(date +%s)

emit_event "{\"event\":\"frames_extracted\",\"total_frames\":$TOTAL_FRAMES}"

# ── Step 2: geometry init + 3DGS training ────────────────────────────────────
echo ""
echo "[2/3] SfM + 3DGS training (sfm=$SFM trainer=$TRAINER, $ITERS iters, $TOTAL_FRAMES frames)..."
mkdir -p "$MODEL_DIR"
cd "$REPO"

SFM_START=$(date +%s)
if [[ "$SFM" == "fast3r" ]]; then
    INIT_GEO_ARGS=""
    [[ -n "$MAX_INIT_POINTS" ]] && INIT_GEO_ARGS="--max_init_points $MAX_INIT_POINTS"
    [[ "$COLMAP_BA"   == "1" ]] && INIT_GEO_ARGS="$INIT_GEO_ARGS --colmap_ba"
    [[ "$NO_POINT_CAP" == "1" ]] && INIT_GEO_ARGS="$INIT_GEO_ARGS --no_point_cap"
    echo "[2/3] Fast3R init + 3DGS training ($ITERS iterations, $TOTAL_FRAMES frames, colmap_ba=$COLMAP_BA)..."
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -W ignore ./init_geo_fast3r.py \
        -s "$SCENE_DIR" \
        -m "$MODEL_DIR" \
        --n_views "$TOTAL_FRAMES" \
        --image_size "$IMAGE_SIZE" \
        --co_vis_dsp \
        $INIT_GEO_ARGS \
        2>&1 | tee "$MODEL_DIR/01_init_geo.log"
elif [[ "$SFM" == "colmap" ]]; then
    echo "[2/3] COLMAP features + matching + incremental SfM ($TOTAL_FRAMES frames)..."
    DB_PATH="$SCENE_DIR/database.db"
    SPARSE_PARENT="$SCENE_DIR/sparse"
    rm -rf "$SPARSE_PARENT" && mkdir -p "$SPARSE_PARENT"
    rm -f "$DB_PATH"
    # /usr/bin/colmap is a Qt GUI binary — needs offscreen platform in headless env
    export QT_QPA_PLATFORM=offscreen

    /usr/bin/colmap feature_extractor \
        --database_path "$DB_PATH" \
        --image_path "$IMAGE_DIR" \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --SiftExtraction.use_gpu 0 \
        2>&1 | tee "$MODEL_DIR/01a_colmap_features.log"

    # Resolve matcher: explicit flag > auto (exhaustive ≤50 frames, sequential >50)
    _MATCHER="${COLMAP_MATCHER}"
    if [[ -z "$_MATCHER" ]]; then
        (( TOTAL_FRAMES <= 50 )) && _MATCHER="exhaustive" || _MATCHER="sequential"
    fi
    echo "    Matcher: $_MATCHER"
    if [[ "$_MATCHER" == "exhaustive" ]]; then
        /usr/bin/colmap exhaustive_matcher \
            --database_path "$DB_PATH" \
            --SiftMatching.use_gpu 0 \
            2>&1 | tee "$MODEL_DIR/01b_colmap_match.log"
    elif [[ "$_MATCHER" == "vocab_tree" ]]; then
        VOCAB_TREE="$REPO/assets/vocab_tree_flickr100K_words32K.bin"
        if [[ ! -f "$VOCAB_TREE" ]]; then
            echo "Error: vocab tree not found at $VOCAB_TREE"
            echo "Download with: wget -O $VOCAB_TREE https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin"
            exit 1
        fi
        /usr/bin/colmap vocab_tree_matcher \
            --database_path "$DB_PATH" \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE" \
            --SiftMatching.use_gpu 0 \
            2>&1 | tee "$MODEL_DIR/01b_colmap_match.log"
    else
        /usr/bin/colmap sequential_matcher \
            --database_path "$DB_PATH" \
            --SequentialMatching.overlap 10 \
            --SiftMatching.use_gpu 0 \
            2>&1 | tee "$MODEL_DIR/01b_colmap_match.log"
    fi

    /usr/bin/colmap mapper \
        --database_path "$DB_PATH" \
        --image_path "$IMAGE_DIR" \
        --output_path "$SPARSE_PARENT" \
        2>&1 | tee "$MODEL_DIR/01c_colmap_sfm.log"

    if [[ ! -d "$SPARSE_PARENT/0" ]]; then
        echo "Error: COLMAP produced no reconstruction. Check $MODEL_DIR/01c_colmap_sfm.log"
        exit 1
    fi

    N_COMPONENTS=$(ls -d "$SPARSE_PARENT"/[0-9]* 2>/dev/null | wc -l)
    (( N_COMPONENTS > 1 )) && echo "    ⚠  $N_COMPONENTS disconnected components — using component 0 (largest)"

    # Outlier filter (needs binary reconstruction; works before text conversion)
    SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
        "$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1 | tee "$MODEL_DIR/01d_colmap_filter.log"

    # Convert binary → text (cameras.txt, images.txt, points3D.txt) for train.py
    /usr/bin/colmap model_converter \
        --input_path "$SPARSE_PARENT/0" \
        --output_path "$SPARSE_PARENT/0" \
        --output_type TXT \
        2>&1 | tee "$MODEL_DIR/01e_colmap_convert.log"
else
    # Clear any cache left by a previous crashed run; stale SparseGA cache causes
    # device-side assert when tensor shapes no longer match the new image set.
    rm -rf "$MODEL_DIR/sparse_ga_cache"
    INIT_GEO_ARGS=""
    [[ -n "$MAX_INIT_POINTS" ]]  && INIT_GEO_ARGS="--max_init_points $MAX_INIT_POINTS"
    [[ "$SPARSE_PAIRS"  == "1" ]] && INIT_GEO_ARGS="$INIT_GEO_ARGS --sparse_pairs"
    [[ "$SPARSE_GA"     == "1" ]] && INIT_GEO_ARGS="$INIT_GEO_ARGS --sparse_ga"
    [[ "$NO_POINT_CAP"  == "1" ]] && INIT_GEO_ARGS="$INIT_GEO_ARGS --no_point_cap"
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
fi
SFM_END=$(date +%s)
SFM_ELAPSED=$(( SFM_END - SFM_START ))
SFM_MIN=$(awk "BEGIN {printf \"%.1f\", $SFM_ELAPSED / 60}")
echo "    → SfM step: ${SFM_ELAPSED}s (${SFM_MIN} min)"

emit_event "{\"event\":\"sfm_done\",\"sfm\":\"$SFM\",\"elapsed_s\":$SFM_ELAPSED}"

if [[ "$SFM" == "mast3r" ]] && (( SFM_ELAPSED > 120 )); then
    echo "    ⚠  Slow MASt3R (>${SFM_ELAPSED}s) — compiling the RoPE2D CUDA kernel would save ~20-40% here. See InstantSplat/TODO.md."
fi

EARLY_STOP_ARGS=""
if [[ "$EARLY_STOP" == "1" ]]; then
    # Smart-frames scenes cover more of the wall — far cameras need more iters
    # to converge, so double the patience to avoid stopping too early.
    PATIENCE=200
    [[ "$SMART_FRAMES" == "1" ]] && PATIENCE=400
    EARLY_STOP_ARGS="--early_stop_patience $PATIENCE --early_stop_delta 1e-5"
fi

TRAIN_ARGS=""
[[ "$NO_DENSIFICATION" == "1" ]] && TRAIN_ARGS="--densify_until_iter 0"

# Log available resources before training
RAM_AVAIL=$(free -h | awk '/^Mem:/ {print $7}')
VRAM_USED=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%d/%d MiB", $1, $2}')
echo "    RAM available: $RAM_AVAIL  |  VRAM: $VRAM_USED"

if [[ "$TRAINER" == "splatfacto" ]]; then
    echo "[2/3] nerfstudio splatfacto training ($ITERS iterations, $TOTAL_FRAMES frames)..."
    NS_PROCESS_DIR="$MODEL_DIR/ns_input"
    NS_TRAIN_DIR="$MODEL_DIR/ns_train"
    SPARSE_PARENT="$SCENE_DIR/sparse"
    ns-process-data images \
        --data "$IMAGE_DIR" \
        --output-dir "$NS_PROCESS_DIR" \
        --skip-colmap \
        --colmap-model-path "$SPARSE_PARENT/0" \
        2>&1 | tee "$MODEL_DIR/02a_ns_process.log"
    ns-train splatfacto \
        --data "$NS_PROCESS_DIR" \
        --output-dir "$NS_TRAIN_DIR" \
        --max-num-iterations "$ITERS" \
        --vis tensorboard \
        2>&1 | tee "$MODEL_DIR/02_train.log"
elif [[ "$TRAINER" == "pgsr" ]]; then
    # For mast3r/fast3r, symlink sparse → sparse_N so sparse/0/ exists.
    [[ "$SFM" != "colmap" ]] && ln -sfn "sparse_${TOTAL_FRAMES}" "$SCENE_DIR/sparse" 2>/dev/null || true
    # PGSR looks for sparse/images.bin (no 0/ subdir) — symlink files up from sparse/0/
    for _f in cameras.txt images.txt points3D.txt cameras.bin images.bin points3D.bin; do
        [[ -f "$SCENE_DIR/sparse/0/$_f" ]] && \
            ln -sfn "0/$_f" "$SCENE_DIR/sparse/$_f" 2>/dev/null || true
    done
    PGSR_DIR="${PGSR_DIR:-/home/communications/workdir/pgsr}"
    PGSR_PYTHON="${PGSR_PYTHON:-$PYTHON}"
    echo "[2/3] PGSR training ($ITERS iterations, $TOTAL_FRAMES frames)..."
    cd "$PGSR_DIR"
    CUDA_VISIBLE_DEVICES=0 "$PGSR_PYTHON" ./train.py \
        -s "$SCENE_DIR" \
        -m "$MODEL_DIR" \
        --iterations "$ITERS" \
        --save_iterations "$ITERS" \
        --test_iterations "$ITERS" \
        2>&1 | tee "$MODEL_DIR/02_train.log"
elif [[ "$TRAINER" == "gsplat" ]]; then
    # gsplat via InstantSplat/simple_trainer.py — already proven on this 8GB machine
    [[ "$SFM" != "colmap" ]] && ln -sfn "sparse_${TOTAL_FRAMES}" "$SCENE_DIR/sparse" 2>/dev/null || true
    GSPLAT_OUT="$MODEL_DIR/gsplat_output"
    echo "[2/3] gsplat training ($ITERS iterations, $TOTAL_FRAMES frames)..."
    PYTHONPATH="$REPO/gsplat_examples" CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/simple_trainer.py" default \
        --data-dir "$SCENE_DIR" \
        --data-factor 1 \
        --result-dir "$GSPLAT_OUT" \
        --max-steps "$ITERS" \
        --save-steps "$ITERS" \
        --ply-steps "$ITERS" \
        --save-ply \
        --init-type sfm \
        --fast-init \
        --disable-viewer \
        --disable-video \
        --ssim-lambda 0.2 \
        2>&1 | tee "$MODEL_DIR/02_train.log"
else
    # instantsplat trainer
    # --pp_optimizer requires confidence_dsp.npy from init_geo.py (MASt3R/Fast3R only)
    PP_OPT_ARG="--pp_optimizer"
    [[ "$SFM" == "colmap" ]] && PP_OPT_ARG=""
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" ./train.py \
        -s "$SCENE_DIR" \
        -m "$MODEL_DIR" \
        -r 1 \
        --n_views "$TOTAL_FRAMES" \
        --iterations "$ITERS" \
        $PP_OPT_ARG \
        --optim_pose \
        $TRAIN_ARGS \
        $EARLY_STOP_ARGS \
        2>&1 | tee "$MODEL_DIR/02_train.log"
fi

echo "    → done"

# ── Emit train_done event (grep tee'd log for early-stop info) ───────────────
TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$(( TRAIN_END - SFM_END ))
EARLY_STOPPED_FLAG="false"
STOPPED_AT_ITER="null"
if grep -q 'Early stopping' "$MODEL_DIR/02_train.log" 2>/dev/null; then
    EARLY_STOPPED_FLAG="true"
    STOPPED_AT_ITER=$(grep -oP '\[ITER \K\d+(?=\] Early stopping)' "$MODEL_DIR/02_train.log" | tail -1)
    STOPPED_AT_ITER=${STOPPED_AT_ITER:-null}
fi
emit_event "{\"event\":\"train_done\",\"elapsed_s\":$TRAIN_ELAPSED,\"early_stopped\":$EARLY_STOPPED_FLAG,\"stopped_at_iter\":$STOPPED_AT_ITER}"

# Resolve PLY path — differs by trainer
if [[ "$TRAINER" == "gsplat" ]]; then
    PLY=$(find "$MODEL_DIR/gsplat_output/ply" -name "*.ply" 2>/dev/null | sort | tail -1)
    ACTUAL_ITER=$ITERS
elif [[ "$TRAINER" == "splatfacto" ]]; then
    NS_CONFIG=$(find "$MODEL_DIR/ns_train" -name "config.yml" 2>/dev/null | sort | tail -1)
    NS_EXPORT_DIR="$MODEL_DIR/ns_export"
    mkdir -p "$NS_EXPORT_DIR"
    ns-export gaussian-splat \
        --load-config "$NS_CONFIG" \
        --output-dir "$NS_EXPORT_DIR" \
        2>&1 | tee "$MODEL_DIR/02b_ns_export.log"
    PLY=$(find "$NS_EXPORT_DIR" -name "*.ply" 2>/dev/null | head -1)
    ACTUAL_ITER=$ITERS
else
    ACTUAL_ITER=$(ls -d "$MODEL_DIR/point_cloud/iteration_"* 2>/dev/null \
        | grep -oP 'iteration_\K\d+' | sort -n | tail -1)
    ACTUAL_ITER=${ACTUAL_ITER:-$ITERS}
    PLY="$MODEL_DIR/point_cloud/iteration_${ACTUAL_ITER}/point_cloud.ply"
fi
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
  "sfm":              "$SFM",
  "trainer":          "$TRAINER",
  "sfm_seconds":      $SFM_ELAPSED,
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
