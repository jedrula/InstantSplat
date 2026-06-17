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
# Skip SfM entirely (pre-posed export: transforms.json + pointcloud.ply + images/ already present):
#   bash video_to_splat.sh --sfm preposed --preposed-dir /path/to/export --scene NAME --iters N --trainer TRAINER
#   (supported trainers: brush, gsplat, 2dgs, splatfacto)
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
CONDA_BIN="$(dirname "$PYTHON")"
ONTHEFLY_PYTHON="${ONTHEFLY_PYTHON:-${HOME}/miniconda3/envs/onthefly_nvs/bin/python}"
ONTHEFLY_REPO="${ONTHEFLY_REPO:-${HOME}/workdir/on-the-fly-nvs}"

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
SFM="mast3r"       # mast3r | fast3r | colmap_sift | glomap_sift | glomap_aliked | glomap_disk | glomap_superpoint | glomap_loftr | colmap_aliked | fastmap | realityscan | onthefly
TRAINER="instantsplat"  # instantsplat | pgsr | splatfacto | gsplat | onthefly | brush
MCMC=0              # 1 = use MCMCStrategy (gsplat only); 0 = DefaultStrategy+absgrad
GSPLAT_POST_PROCESSING=""   # "" | bilateral_grid | ppisp
BILATERAL_GRID_FUSED=0  # 1 = use fused bilateral grid impl (requires fused_bilagrid); only with bilateral_grid
RANDOM_BKGD=0           # 1 = randomize background color during training (helps unbounded scenes)
GSPLAT_SSIM_LAMBDA="0.2"
VIEWER_PORT=""      # empty = disable viewer; set to a port number to enable viser viewer
ONTHEFLY_ITERS=30       # per-keyframe iterations for on-the-fly NVS (--sfm onthefly)
COLMAP_BA=0
COLMAP_MATCHER=""
VIEW_GRAPH_CALIBRATOR=0
NO_POINT_CAP=0
NO_DENSIFICATION=0
FRAMES_ONLY=0
SKIP_EXTRACTION=0
IMAGE_SIZE=256
MODEL_DIR_OVERRIDE=""
PREPOSED_DIR=""
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
        --mcmc)             MCMC=1;              shift ;;
        --post-processing)  GSPLAT_POST_PROCESSING="$2"; shift 2 ;;
        --bilateral-grid-fused) BILATERAL_GRID_FUSED=1; shift ;;
        --random-bkgd)      RANDOM_BKGD=1;       shift ;;
        --ssim-lambda)      GSPLAT_SSIM_LAMBDA="$2";     shift 2 ;;
        --viewer-port)      VIEWER_PORT="$2";    shift 2 ;;
        --engine)           # deprecated: map to --sfm + --trainer
            case "$2" in
                pgsr)    SFM="mast3r";  TRAINER="pgsr" ;;
                fast3r)  SFM="fast3r";  TRAINER="instantsplat" ;;
                colmap)  SFM="colmap_sift";  TRAINER="instantsplat" ;;
                glomap)  SFM="glomap_sift";  TRAINER="instantsplat" ;;
                glomap_aliked)    SFM="glomap_aliked";    TRAINER="instantsplat" ;;
                glomap_disk)  SFM="glomap_disk";  TRAINER="instantsplat" ;;
                glomap_superpoint)    SFM="glomap_superpoint";    TRAINER="instantsplat" ;;
                glomap_loftr) SFM="glomap_loftr"; TRAINER="instantsplat" ;;
                colmap_aliked)    SFM="colmap_aliked";    TRAINER="instantsplat" ;;
                fastmap)      SFM="fastmap";      TRAINER="instantsplat" ;;
                realityscan)  SFM="realityscan"; TRAINER="instantsplat" ;;
                onthefly)     SFM="onthefly";    TRAINER="onthefly" ;;
                *)            SFM="mast3r";      TRAINER="instantsplat" ;;
            esac
            shift 2 ;;
        --colmap-ba)        COLMAP_BA=1;         shift ;;
        --colmap-matcher)   COLMAP_MATCHER="$2"; shift 2 ;;
        --view-graph-calibrator) VIEW_GRAPH_CALIBRATOR=1; shift ;;
        --no-point-cap)      NO_POINT_CAP=1;      shift ;;
        --no-densification)  NO_DENSIFICATION=1;  shift ;;
        --image-size)   IMAGE_SIZE="$2";   shift 2 ;;
        --model-dir)    MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
        --preposed-dir) PREPOSED_DIR="$2";      shift 2 ;;
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

# ── Preposed: validate inputs + skip extraction ───────────────────────────────
if [[ "$SFM" == "preposed" ]]; then
    [[ -z "$PREPOSED_DIR" ]] && { echo "Error: --sfm preposed requires --preposed-dir PATH"; exit 1; }
    [[ ! -d "$PREPOSED_DIR" ]] && { echo "Error: --preposed-dir not found: $PREPOSED_DIR"; exit 1; }
    [[ ! -f "$PREPOSED_DIR/transforms.json" ]] && { echo "Error: $PREPOSED_DIR/transforms.json not found"; exit 1; }
    [[ ! -d "$PREPOSED_DIR/images" ]] && { echo "Error: $PREPOSED_DIR/images not found"; exit 1; }
    SKIP_EXTRACTION=1
    case "$TRAINER" in
        brush|gsplat|2dgs|splatfacto) ;;
        *) echo "Error: --sfm preposed does not support --trainer $TRAINER (supported: brush, gsplat, 2dgs, splatfacto)"; exit 1 ;;
    esac
fi

[[ "$SKIP_EXTRACTION" != "1" && ${#VIDEOS[@]} -eq 0 ]] && { echo "$USAGE"; exit 1; }
[[ -z "$SCENE" ]]          && { echo "Error: --scene is required"; echo "$USAGE"; exit 1; }
[[ -z "$ITERS" && "$FRAMES_ONLY" != "1" && "$SFM" != "onthefly" ]] && { echo "Error: --iters is required"; echo "$USAGE"; exit 1; }
ITERS="${ITERS:-0}"
[[ "$SFM" == "onthefly" ]] && TRAINER="onthefly"

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

# For preposed, the input dir already has transforms.json + images/ + PLY — use it directly
if [[ "$SFM" == "preposed" ]]; then
    SCENE_DIR="$PREPOSED_DIR"
    IMAGE_DIR="$SCENE_DIR/images"
fi

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
elif [[ "$SFM" == "colmap_sift" ]]; then
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
elif [[ "$SFM" == "glomap_sift" ]]; then
    echo "[2/3] COLMAP features + matching + GLOMAP global SfM ($TOTAL_FRAMES frames)..."
    DB_PATH="$SCENE_DIR/database.db"
    SPARSE_PARENT="$SCENE_DIR/sparse"
    rm -rf "$SPARSE_PARENT" && mkdir -p "$SPARSE_PARENT"
    rm -f "$DB_PATH"
    export QT_QPA_PLATFORM=offscreen

    # GLOMAP is deprecated upstream; its global mapper is now part of COLMAP 4.x.
    # Use $CONDA_BIN/colmap throughout so features, matching, and global_mapper
    # all share the same DB schema — no version mismatch, retriangulation works.
    "$CONDA_BIN/colmap" feature_extractor \
        --database_path "$DB_PATH" \
        --image_path "$IMAGE_DIR" \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --FeatureExtraction.use_gpu 1 \
        --FeatureExtraction.max_image_size 1600 \
        2>&1 | tee "$MODEL_DIR/01a_glomap_features.log"

    _MATCHER="${COLMAP_MATCHER}"
    if [[ -z "$_MATCHER" ]]; then
        (( TOTAL_FRAMES <= 50 )) && _MATCHER="exhaustive" || _MATCHER="sequential"
    fi
    echo "    Matcher: $_MATCHER"
    if [[ "$_MATCHER" == "exhaustive" ]]; then
        "$CONDA_BIN/colmap" exhaustive_matcher \
            --database_path "$DB_PATH" \
            --FeatureMatching.use_gpu 1 \
            2>&1 | tee "$MODEL_DIR/01b_glomap_match.log"
    elif [[ "$_MATCHER" == "vocab_tree" ]]; then
        # COLMAP 4.x uses faiss (not flann) — needs the faiss-format tree
        VOCAB_TREE="$REPO/assets/vocab_tree_faiss_flickr100K_words256K.bin"
        if [[ ! -f "$VOCAB_TREE" ]]; then
            echo "Error: vocab tree not found at $VOCAB_TREE"
            echo "Download with: wget -O $VOCAB_TREE https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
            exit 1
        fi
        "$CONDA_BIN/colmap" vocab_tree_matcher \
            --database_path "$DB_PATH" \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE" \
            --FeatureMatching.use_gpu 1 \
            2>&1 | tee "$MODEL_DIR/01b_glomap_match.log"
    else
        "$CONDA_BIN/colmap" sequential_matcher \
            --database_path "$DB_PATH" \
            --SequentialMatching.overlap 10 \
            --FeatureMatching.use_gpu 1 \
            2>&1 | tee "$MODEL_DIR/01b_glomap_match.log"
    fi

    if [[ "$VIEW_GRAPH_CALIBRATOR" == "1" ]]; then
        echo "    Running view_graph_calibrator (focal length estimation)..."
        "$CONDA_BIN/colmap" view_graph_calibrator \
            --database_path "$DB_PATH" \
            --image_path "$IMAGE_DIR" \
            2>&1 | tee "$MODEL_DIR/01b2_glomap_vgc.log"
    fi

    "$CONDA_BIN/colmap" global_mapper \
        --database_path "$DB_PATH" \
        --image_path "$IMAGE_DIR" \
        --output_path "$SPARSE_PARENT" \
        2>&1 | tee "$MODEL_DIR/01c_glomap_sfm.log"

    if [[ ! -d "$SPARSE_PARENT/0" ]]; then
        echo "Error: GLOMAP produced no reconstruction. Check $MODEL_DIR/01c_glomap_sfm.log"
        exit 1
    fi

    SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
        "$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1 | tee "$MODEL_DIR/01d_glomap_filter.log"

    "$CONDA_BIN/colmap" model_converter \
        --input_path "$SPARSE_PARENT/0" \
        --output_path "$SPARSE_PARENT/0" \
        --output_type TXT \
        2>&1 | tee "$MODEL_DIR/01e_glomap_convert.log"

    # Persist database for query-image localization.
    # 1. Remap image IDs to match sparse model (GLOMAP reassigns them vs feature_extractor).
    # 2. Expand images.txt to full COLMAP format (GLOMAP only stores matched keypoints;
    #    image_registrator requires ALL keypoints listed with -1 for unmatched ones).
    if [[ -f "$DB_PATH" && -f "$SPARSE_PARENT/0/images.txt" ]]; then
        python3 "$REPO/remap_db_to_sparse.py" \
            "$DB_PATH" "$SPARSE_PARENT/0/images.txt" "$MODEL_DIR/database.db"
        python3 "$REPO/expand_glomap_images.py" \
            "$MODEL_DIR/database.db" "$SPARSE_PARENT/0" --in-place
    fi

elif [[ "$SFM" == "glomap_aliked" ]]; then
    echo "[2/3] ALIKED+LightGlue features + GLOMAP global SfM ($TOTAL_FRAMES frames)..."
    SPARSE_PARENT="$SCENE_DIR/sparse"
    rm -rf "$SPARSE_PARENT" && mkdir -p "$SPARSE_PARENT"
    HLOC_WORK="$SCENE_DIR/hloc_work"
    rm -rf "$HLOC_WORK"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/glomap_hloc.py" \
        "$IMAGE_DIR" \
        "$HLOC_WORK" \
        --matcher aliked+lightglue \
        --colmap-bin "$CONDA_BIN/colmap" \
        2>&1 | tee "$MODEL_DIR/01_glomap_aliked.log"

    if [[ ! -d "$HLOC_WORK/sparse/0" ]]; then
        echo "Error: glomap_hloc produced no reconstruction. Check $MODEL_DIR/01_glomap_aliked.log"
        exit 1
    fi

    mv "$HLOC_WORK/sparse/0" "$SPARSE_PARENT/0"

    SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
        "$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1 | tee "$MODEL_DIR/01b_glomap_aliked_filter.log"

    "$CONDA_BIN/colmap" model_converter \
        --input_path "$SPARSE_PARENT/0" \
        --output_path "$SPARSE_PARENT/0" \
        --output_type TXT \
        2>&1 | tee "$MODEL_DIR/01c_glomap_aliked_convert.log"
elif [[ "$SFM" == "glomap_loftr" ]]; then
    echo "[2/3] LoFTR semi-dense matching + GLOMAP global SfM ($TOTAL_FRAMES frames)..."
    SPARSE_PARENT="$SCENE_DIR/sparse"
    rm -rf "$SPARSE_PARENT" && mkdir -p "$SPARSE_PARENT"
    HLOC_WORK="$SCENE_DIR/hloc_work"
    rm -rf "$HLOC_WORK"

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/glomap_hloc.py" \
        "$IMAGE_DIR" \
        "$HLOC_WORK" \
        --matcher loftr_indoor \
        --colmap-bin "$CONDA_BIN/colmap" \
        2>&1 | tee "$MODEL_DIR/01_glomap_loftr.log"

    if [[ ! -d "$HLOC_WORK/sparse/0" ]]; then
        echo "Error: glomap_hloc (loftr) produced no reconstruction. Check $MODEL_DIR/01_glomap_loftr.log"
        exit 1
    fi

    mv "$HLOC_WORK/sparse/0" "$SPARSE_PARENT/0"

    SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
        "$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1 | tee "$MODEL_DIR/01b_glomap_loftr_filter.log"

    "$CONDA_BIN/colmap" model_converter \
        --input_path "$SPARSE_PARENT/0" \
        --output_path "$SPARSE_PARENT/0" \
        --output_type TXT \
        2>&1 | tee "$MODEL_DIR/01c_glomap_loftr_convert.log"
elif [[ "$SFM" == "glomap_disk" || "$SFM" == "glomap_superpoint" || "$SFM" == "colmap_aliked" ]]; then
    case "$SFM" in
        glomap_disk) _MATCHER="disk+lightglue";       _MAPPER="glomap" ;;
        glomap_superpoint)   _MATCHER="superpoint+lightglue"; _MAPPER="glomap" ;;
        colmap_aliked)   _MATCHER="aliked+lightglue";     _MAPPER="colmap" ;;
    esac
    echo "[2/3] hloc($_MATCHER) + $_MAPPER SfM ($TOTAL_FRAMES frames)..."
    SPARSE_PARENT="$SCENE_DIR/sparse"
    rm -rf "$SPARSE_PARENT" && mkdir -p "$SPARSE_PARENT"
    HLOC_WORK="$SCENE_DIR/hloc_work"
    rm -rf "$HLOC_WORK"

    # Focal length prior from previous reconstruction if available (fixes GLOMAP warning)
    _FL_ARG=""
    _PREV_TF=$(find "$SCENE_DIR" -name "transforms.json" 2>/dev/null | head -1)
    if [[ -n "$_PREV_TF" ]]; then
        _FL=$(python3 -c "import json; d=json.load(open('$_PREV_TF')); print(d.get('fl_x',''))" 2>/dev/null || true)
        [[ -n "$_FL" ]] && _FL_ARG="--focal-length $_FL"
    fi

    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/glomap_hloc.py" \
        "$IMAGE_DIR" \
        "$HLOC_WORK" \
        --matcher "$_MATCHER" \
        --mapper "$_MAPPER" \
        $_FL_ARG \
        --colmap-bin "$CONDA_BIN/colmap" \
        2>&1 | tee "$MODEL_DIR/01_${SFM}.log"

    if [[ ! -d "$HLOC_WORK/sparse/0" ]]; then
        echo "Error: hloc SfM produced no reconstruction. Check $MODEL_DIR/01_${SFM}.log"
        exit 1
    fi

    mv "$HLOC_WORK/sparse/0" "$SPARSE_PARENT/0"

    SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
        "$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1 | tee "$MODEL_DIR/01b_${SFM}_filter.log"

    "$CONDA_BIN/colmap" model_converter \
        --input_path "$SPARSE_PARENT/0" \
        --output_path "$SPARSE_PARENT/0" \
        --output_type TXT \
        2>&1 | tee "$MODEL_DIR/01c_${SFM}_convert.log"
elif [[ "$SFM" == "fastmap" ]]; then
    echo "[2/3] COLMAP features + matching + FastMap pose estimation ($TOTAL_FRAMES frames)..."
    DB_PATH="$SCENE_DIR/database.db"
    SPARSE_PARENT="$SCENE_DIR/sparse"
    FM_OUTPUT="$SCENE_DIR/fastmap_out"
    rm -rf "$SPARSE_PARENT" && mkdir -p "$SPARSE_PARENT"
    rm -f "$DB_PATH"
    rm -rf "$FM_OUTPUT"
    export QT_QPA_PLATFORM=offscreen

    "$CONDA_BIN/colmap" feature_extractor \
        --database_path "$DB_PATH" \
        --image_path "$IMAGE_DIR" \
        --ImageReader.camera_model PINHOLE \
        --ImageReader.single_camera 1 \
        --FeatureExtraction.use_gpu 1 \
        --FeatureExtraction.max_image_size 1600 \
        2>&1 | tee "$MODEL_DIR/01a_fastmap_features.log"

    _MATCHER="${COLMAP_MATCHER}"
    if [[ -z "$_MATCHER" ]]; then
        (( TOTAL_FRAMES <= 50 )) && _MATCHER="exhaustive" || _MATCHER="sequential"
    fi
    echo "    Matcher: $_MATCHER"
    if [[ "$_MATCHER" == "exhaustive" ]]; then
        "$CONDA_BIN/colmap" exhaustive_matcher \
            --database_path "$DB_PATH" \
            --FeatureMatching.use_gpu 1 \
            2>&1 | tee "$MODEL_DIR/01b_fastmap_match.log"
    elif [[ "$_MATCHER" == "vocab_tree" ]]; then
        VOCAB_TREE="$REPO/assets/vocab_tree_faiss_flickr100K_words256K.bin"
        if [[ ! -f "$VOCAB_TREE" ]]; then
            echo "Error: vocab tree not found at $VOCAB_TREE"
            echo "Download with: wget -O $VOCAB_TREE https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
            exit 1
        fi
        "$CONDA_BIN/colmap" vocab_tree_matcher \
            --database_path "$DB_PATH" \
            --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE" \
            --FeatureMatching.use_gpu 1 \
            2>&1 | tee "$MODEL_DIR/01b_fastmap_match.log"
    else
        "$CONDA_BIN/colmap" sequential_matcher \
            --database_path "$DB_PATH" \
            --SequentialMatching.overlap 10 \
            --FeatureMatching.use_gpu 1 \
            2>&1 | tee "$MODEL_DIR/01b_fastmap_match.log"
    fi

    FASTMAP_DIR="${FASTMAP_DIR:-/home/communications/workdir/fastmap}"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$FASTMAP_DIR/run.py" \
        --database "$DB_PATH" \
        --image_dir "$IMAGE_DIR" \
        --output_dir "$FM_OUTPUT" \
        --pinhole \
        --headless \
        2>&1 | tee "$MODEL_DIR/01c_fastmap_sfm.log"

    if [[ ! -d "$FM_OUTPUT/sparse/0" ]]; then
        echo "Error: FastMap produced no reconstruction. Check $MODEL_DIR/01c_fastmap_sfm.log"
        exit 1
    fi

    mv "$FM_OUTPUT/sparse/0" "$SPARSE_PARENT/0"

    SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
        "$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1 | tee "$MODEL_DIR/01d_fastmap_filter.log"

    /usr/bin/colmap model_converter \
        --input_path "$SPARSE_PARENT/0" \
        --output_path "$SPARSE_PARENT/0" \
        --output_type TXT \
        2>&1 | tee "$MODEL_DIR/01e_fastmap_convert.log"
elif [[ "$SFM" == "realityscan" ]]; then
    echo "[2/3] RealityScan alignment + COLMAP export ($TOTAL_FRAMES frames)..."
    # RealityScan always needs an X display even with -stdConsole; xvfb-run -a
    # spins up a throwaway virtual display (forum: unrealengine.com
    # /t/realityscan-fully-headless-linux/2682217).
    # Must call wine directly with --cx-app (CreateProcessW) rather than via
    # realityscan-cli, which uses ShellExecuteExW and concatenates exe+args
    # into lpFile causing "cannot execute" error.
    # Z: drive maps to Linux root (/).
    RS_WINE="${REALITYSCAN_WINE:-/opt/realityscan/bin/wine}"
    RS_EXE="C:/Program Files/Epic Games/RealityScan/RealityScan.exe"
    RS_OUT="$SCENE_DIR/rs_output"
    SPARSE_PARENT="$SCENE_DIR/sparse"

    if [[ ! -x "$RS_WINE" ]]; then
        echo "Error: RealityScan wine launcher not found at $RS_WINE"
        echo "  Install: sudo apt install ~/Downloads/RealityScan-*.deb"
        echo "  Or set:  REALITYSCAN_WINE=/opt/realityscan/bin/wine"
        exit 1
    fi
    if ! command -v xvfb-run &>/dev/null; then
        echo "Error: xvfb-run not found — install it: sudo apt install xvfb"
        exit 1
    fi

    rm -rf "$RS_OUT" "$SPARSE_PARENT"
    mkdir -p "$RS_OUT" "$SPARSE_PARENT/0"

    # RealityScan (Wine) cannot decode progressive JPEGs — re-encode them as baseline in-place.
    # SOF2 marker (0xFFC2) = progressive; SOF0 (0xFFC0) = baseline.
    python3 - "$IMAGE_DIR" <<'PYEOF'
import sys, os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
img_dir = sys.argv[1]
for fname in os.listdir(img_dir):
    if not fname.lower().endswith(('.jpg', '.jpeg')):
        continue
    path = os.path.join(img_dir, fname)
    with open(path, 'rb') as f:
        data = f.read(65536)
    if b'\xff\xc2' in data:  # SOF2 = progressive JPEG
        print(f"    ⚠  re-encoding progressive JPEG: {fname}")
        im = Image.open(path).convert('RGB')
        im.save(path, 'JPEG', quality=95, progressive=False, optimize=False)
PYEOF

    # rs_colmap_params.xml  → COLMAP writer (images.txt + points3D.txt, undistorted images)
    # rs_csv_params.xml     → CSV intrinsics export (needed to synthesise cameras.txt)
    RS_PARAMS="Z:${REPO}/rs_colmap_params.xml"
    RS_CSV_PARAMS="Z:${REPO}/rs_csv_params.xml"

    # COLMAP writer puts output in RS_OUT/sparse/0/  (COLMAP standard layout).
    # CSV writer puts intrinsics in RS_OUT/intrinsics.csv.
    # 300 s timeout: RS should finish alignment in < 2 min; a hang means Epic auth
    # failed (expired session). Kill-on-timeout lets the pipeline fail fast.
    RS_TIMEOUT="${REALITYSCAN_TIMEOUT:-300}"
    timeout "$RS_TIMEOUT" xvfb-run -a "$RS_WINE" \
        --bottle=default \
        --cx-app "$RS_EXE" \
        -- \
        -stdConsole \
        -set appIgnoreExifGPS=true \
        -set sfmEnableCameraPrior=false \
        -addFolder "Z:${IMAGE_DIR}" \
        -align \
        -selectMaximalComponent \
        -exportRegistration "Z:${RS_OUT}/cameras.txt" "$RS_PARAMS" \
        -exportRegistration "Z:${RS_OUT}/intrinsics.csv" "$RS_CSV_PARAMS" \
        -exportSparsePointCloud "Z:${RS_OUT}/points3D.ply" \
        -quit \
        2>&1 | tee "$MODEL_DIR/01a_rs_align.log" \
    || { ec=$?; [[ $ec -eq 124 ]] && echo "Error: RealityScan timed out after ${RS_TIMEOUT}s — Epic session may have expired; re-login on the physical machine and retry" || echo "Error: RealityScan exited with code $ec"; exit $ec; }

    # COLMAP writer outputs to RS_OUT/sparse/0/ (standard) or RS_OUT/ (flat).
    RS_COLMAP_DIR="$RS_OUT/sparse/0"
    [[ ! -f "$RS_COLMAP_DIR/images.txt" ]] && RS_COLMAP_DIR="$RS_OUT"

    if [[ ! -f "$RS_COLMAP_DIR/images.txt" ]]; then
        echo "Error: RealityScan produced no images.txt. Check $MODEL_DIR/01a_rs_align.log"
        exit 1
    fi

    # Move images.txt + points3D.txt (cameras.txt is synthesised below).
    for _f in images.txt points3D.txt; do
        [[ -f "$RS_COLMAP_DIR/$_f" ]] && mv "$RS_COLMAP_DIR/$_f" "$SPARSE_PARENT/0/"
    done

    # Synthesise cameras.txt from CSV intrinsics + images.txt.
    RS_UNDIST_DIR="$RS_OUT/images"
    "$PYTHON" "$REPO/rs_make_cameras_txt.py" \
        "$SPARSE_PARENT/0/images.txt" \
        "$RS_OUT/intrinsics.csv" \
        "$RS_UNDIST_DIR" \
        "$SPARSE_PARENT/0/cameras.txt" \
        2>&1 | tee -a "$MODEL_DIR/01a_rs_align.log"

    if [[ ! -f "$SPARSE_PARENT/0/cameras.txt" ]]; then
        echo "Error: rs_make_cameras_txt.py failed. Check $MODEL_DIR/01a_rs_align.log"
        exit 1
    fi

    # RS undistorted images are RGBA and vary in size per image (different undistortion crops).
    # Normalise them to the canonical W×H in cameras.txt (RGB, consistent size) so trainers
    # don't crash on dimension mismatches.
    python3 - "$SPARSE_PARENT/0/cameras.txt" "$RS_UNDIST_DIR" <<'PYEOF'
import sys, os
from PIL import Image

cam_file, img_dir = sys.argv[1], sys.argv[2]
# Read first camera's width/height from cameras.txt
W = H = None
with open(cam_file) as f:
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        W, H = int(parts[2]), int(parts[3])
        break
if W is None:
    print("  ⚠  could not read camera dims — skipping normalisation"); sys.exit(0)
print(f"  normalising RS images → {W}×{H} RGB")
for fname in os.listdir(img_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    path = os.path.join(img_dir, fname)
    im = Image.open(path).convert('RGB')
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    im.save(path, 'PNG')
print(f"  done")
PYEOF

    # Convert text reconstruction to binary so gsplat/pgsr/nerfstudio fast loaders work.
    "$PYTHON" -c "
import pycolmap
from pathlib import Path
p = Path('$SPARSE_PARENT/0')
r = pycolmap.Reconstruction(str(p))
r.write_binary(str(p))
print(f'  → text→binary: {len(r.cameras)} cameras, {len(r.images)} images, {len(r.points3D)} points')
" 2>&1 | tee -a "$MODEL_DIR/01a_rs_align.log"

    # Use the undistorted images for training (camera models are for those images).
    IMAGE_DIR="$RS_UNDIST_DIR"
    TOTAL_FRAMES=$(find "$IMAGE_DIR" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) | wc -l)

    # If points3D is missing (registration-only export), synthesise it from the PLY
    if [[ ! -f "$SPARSE_PARENT/0/points3D.bin" && ! -f "$SPARSE_PARENT/0/points3D.txt" ]]; then
        if [[ -f "$RS_OUT/points3D.ply" ]]; then
            "$PYTHON" "$REPO/rs_ply_to_points3d.py" \
                "$RS_OUT/points3D.ply" \
                "$SPARSE_PARENT/0/points3D.bin" \
                2>&1 | tee -a "$MODEL_DIR/01a_rs_align.log"
        else
            # No points at all — write an empty points3D.txt so pycolmap can load
            printf "# 3D point list\n# Number of points: 0\n" > "$SPARSE_PARENT/0/points3D.txt"
            echo "    ⚠  No sparse points found — initialising with empty points3D"
        fi
    fi

    SPARSE_PATH="$SPARSE_PARENT/0" IMAGE_DIR_PATH="$IMAGE_DIR" \
        "$PYTHON" "$REPO/filter_sfm_outliers.py" 2>&1 | tee "$MODEL_DIR/01b_rs_filter.log"

    # Symlink surviving undistorted images into SCENE_DIR/images/ so trainers find
    # them by their RS names (00000.png, 00001.png, …) via the standard data-dir layout.
    # Run after filter so only non-outlier images get linked.
    find "$RS_UNDIST_DIR" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) | \
        while read -r _img; do ln -sfn "$_img" "$SCENE_DIR/images/$(basename "$_img")"; done
    IMAGE_DIR="$SCENE_DIR/images"

    /usr/bin/colmap model_converter \
        --input_path "$SPARSE_PARENT/0" \
        --output_path "$SPARSE_PARENT/0" \
        --output_type TXT \
        2>&1 | tee "$MODEL_DIR/01c_rs_convert.log"
elif [[ "$SFM" == "preposed" ]]; then
    echo "[2/3] SfM skipped — converting nerfstudio poses → COLMAP sparse from $PREPOSED_DIR"
    SPARSE_PARENT="$MODEL_DIR/sparse"
    mkdir -p "$SPARSE_PARENT/0"
    "$PYTHON" "$REPO/ns_to_colmap.py" \
        "$SCENE_DIR/transforms.json" \
        "$SPARSE_PARENT/0" \
        2>&1 | tee "$MODEL_DIR/01_ns_to_colmap.log"
    if [[ ! -f "$SPARSE_PARENT/0/cameras.bin" ]]; then
        echo "Error: ns_to_colmap.py failed — check $MODEL_DIR/01_ns_to_colmap.log"
        exit 1
    fi
    # Symlink images into MODEL_DIR so brush/gsplat find them when scene_dir=MODEL_DIR
    ln -sfn "$SCENE_DIR/images" "$MODEL_DIR/images" 2>/dev/null || true
    ln -sfn "$SCENE_DIR/images" "$MODEL_DIR/sparse/images" 2>/dev/null || true
    # Also make sparse accessible from SCENE_DIR (needed by some trainer loaders)
    ln -sfn "$SPARSE_PARENT" "$SCENE_DIR/sparse" 2>/dev/null || true
elif [[ "$SFM" == "onthefly" ]]; then
    ONTHEFLY_OUT="$MODEL_DIR/onthefly_out"
    echo "[2/3] On-the-fly NVS — joint SfM+Gaussian training (${ONTHEFLY_ITERS} iters/keyframe)..."
    (cd "$ONTHEFLY_REPO" && "$ONTHEFLY_PYTHON" train.py \
        -s "$SCENE_DIR" \
        --images_dir images \
        -m "$ONTHEFLY_OUT" \
        --num_iterations "$ONTHEFLY_ITERS") \
        2>&1 | tee "$MODEL_DIR/02_onthefly.log"
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
# Export sparse point cloud as PLY for browser preview
if [[ "$SFM" == "colmap_sift" || "$SFM" == "glomap_sift" || "$SFM" == "glomap_aliked" || "$SFM" == "glomap_loftr" || "$SFM" == "glomap_disk" || "$SFM" == "glomap_superpoint" || "$SFM" == "colmap_aliked" || "$SFM" == "fastmap" || "$SFM" == "realityscan" ]]; then
    _PC_SRC="$SCENE_DIR/sparse/0"
else
    _PC_SRC="$SCENE_DIR/sparse_${TOTAL_FRAMES}/0"
fi
if [[ -d "$_PC_SRC" ]]; then
    /usr/bin/colmap model_converter \
        --input_path "$_PC_SRC" \
        --output_path "$MODEL_DIR/sfm_pointcloud.ply" \
        --output_type PLY 2>/dev/null || true
fi

# Copy COLMAP sparse into pod so it is self-contained for LichtFeld / re-training
if [[ "$SFM" == "colmap_sift" || "$SFM" == "glomap_sift" || "$SFM" == "glomap_aliked" || \
      "$SFM" == "glomap_loftr" || "$SFM" == "glomap_disk" || "$SFM" == "glomap_superpoint" || \
      "$SFM" == "colmap_aliked" || "$SFM" == "fastmap" || "$SFM" == "realityscan" ]]; then
    if [[ -d "$SPARSE_PARENT" ]]; then
        cp -r "$SPARSE_PARENT" "$MODEL_DIR/"
        # Thin colmap/ dir so LichtFeld can load without hitting the meta.json SOG check
        mkdir -p "$MODEL_DIR/colmap"
        ln -sfn ../sparse "$MODEL_DIR/colmap/sparse"
        ln -sfn ../images "$MODEL_DIR/colmap/images"
    fi
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

if [[ "$SFM" == "onthefly" ]]; then
    echo "[2/3] On-the-fly: training completed in combined SfM+train step."
elif [[ "$TRAINER" == "splatfacto" ]]; then
    echo "[2/3] nerfstudio splatfacto training ($ITERS iterations, $TOTAL_FRAMES frames)..."
    NS_PROCESS_DIR="$MODEL_DIR/ns_input"
    NS_TRAIN_DIR="$MODEL_DIR/ns_train"
    # MASt3R/Fast3R write sparse_N/0/; classical SfM writes sparse/0/
    if [[ "$SFM" == "mast3r" || "$SFM" == "fast3r" ]]; then
        SPARSE_PARENT="$SCENE_DIR/sparse_${TOTAL_FRAMES}"
    elif [[ "$SFM" == "preposed" ]]; then
        SPARSE_PARENT=""
    else
        SPARSE_PARENT="$SCENE_DIR/sparse"
    fi
    # MASt3R/Fast3R output points3D.ply — convert to points3D.bin for nerfstudio
    if [[ "$SFM" == "mast3r" || "$SFM" == "fast3r" ]]; then
        if [[ ! -f "$SPARSE_PARENT/0/points3D.bin" && -f "$SPARSE_PARENT/0/points3D.ply" ]]; then
            "$PYTHON" "$REPO/rs_ply_to_points3d.py" \
                "$SPARSE_PARENT/0/points3D.ply" \
                "$SPARSE_PARENT/0/points3D.bin" \
                2>&1 | tee -a "$MODEL_DIR/02a_ns_process.log"
        fi
    fi
    if [[ "$SFM" == "preposed" ]]; then
        # transforms.json already present in SCENE_DIR — train directly
        NS_TRAIN_DIR="$MODEL_DIR/ns_train"
        ns-train splatfacto \
            --data "$SCENE_DIR" \
            --output-dir "$NS_TRAIN_DIR" \
            --max-num-iterations "$ITERS" \
            --vis tensorboard \
            2>&1 | tee "$MODEL_DIR/02_train.log"
    else
        # RS exports undistorted images; COLMAP poses are calibrated for those, not originals.
        # Use undistorted images when present (realityscan SfM), else fall back to IMAGE_DIR.
        NS_IMAGE_SRC="$IMAGE_DIR"
        [[ "$SFM" == "realityscan" && -d "$SCENE_DIR/rs_output/images" ]] && \
            NS_IMAGE_SRC="$SCENE_DIR/rs_output/images"
        ns-process-data images \
            --data "$NS_IMAGE_SRC" \
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
        # Extract initial camera from training views for the splat viewer
        [[ -f "$NS_PROCESS_DIR/transforms.json" ]] && \
            python3 "$REPO/camera_from_colmap.py" \
                --sparse   "$SPARSE_PARENT/0" \
                --trainer  splatfacto \
                --ns-process-dir "$NS_PROCESS_DIR" \
                --ns-train-dir   "$NS_TRAIN_DIR" \
                --splat    "$(ls "$MODEL_DIR"/*.splat 2>/dev/null | head -1)" \
                --out      "$MODEL_DIR/initial_camera.json" \
                2>/dev/null || true
    fi
elif [[ "$TRAINER" == "pgsr" ]]; then
    # For mast3r/fast3r, symlink sparse → sparse_N so sparse/0/ exists.
    [[ "$SFM" != "colmap_sift" && "$SFM" != "glomap_sift" && "$SFM" != "glomap_aliked" && "$SFM" != "glomap_loftr" && "$SFM" != "glomap_disk" && "$SFM" != "glomap_superpoint" && "$SFM" != "colmap_aliked" && "$SFM" != "fastmap" && "$SFM" != "realityscan" && "$SFM" != "preposed" ]] && ln -sfn "sparse_${TOTAL_FRAMES}" "$SCENE_DIR/sparse" 2>/dev/null || true
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
    # Extract initial camera (no colmap_to_ply_transform.npy for pgsr yet — COLMAP world space)
    python3 "$REPO/camera_from_colmap.py" \
        --sparse     "$SPARSE_PARENT/0" \
        --trainer    pgsr \
        --result-dir "$MODEL_DIR" \
        --splat      "$(ls "$MODEL_DIR"/*.splat 2>/dev/null | head -1)" \
        --out        "$MODEL_DIR/initial_camera.json" \
        2>/dev/null || true
elif [[ "$TRAINER" == "gsplat" ]]; then
    # gsplat via InstantSplat/simple_trainer.py — already proven on this 8GB machine
    # preposed: force MCMC (default strategy FPEs during densification with dense PLY init)
    [[ "$SFM" == "preposed" ]] && MCMC=1
    [[ "$SFM" != "colmap_sift" && "$SFM" != "glomap_sift" && "$SFM" != "glomap_aliked" && "$SFM" != "glomap_loftr" && "$SFM" != "glomap_disk" && "$SFM" != "glomap_superpoint" && "$SFM" != "colmap_aliked" && "$SFM" != "fastmap" && "$SFM" != "realityscan" && "$SFM" != "preposed" ]] && ln -sfn "sparse_${TOTAL_FRAMES}" "$SCENE_DIR/sparse" 2>/dev/null || true
    GSPLAT_OUT="$MODEL_DIR/gsplat_output"
    echo "[2/3] gsplat training ($ITERS iterations, $TOTAL_FRAMES frames, mcmc=$MCMC, post_processing=${GSPLAT_POST_PROCESSING:-none})..."
    _GSPLAT_PP_ARGS=()
    [[ -n "$GSPLAT_POST_PROCESSING" ]] && _GSPLAT_PP_ARGS=(--post-processing "$GSPLAT_POST_PROCESSING")
    [[ "$BILATERAL_GRID_FUSED" == "1" && "$GSPLAT_POST_PROCESSING" == "bilateral_grid" ]] && _GSPLAT_PP_ARGS+=(--bilateral-grid-fused)
    [[ "$RANDOM_BKGD" == "1" ]] && _GSPLAT_PP_ARGS+=(--random-bkgd)
    if [[ "$MCMC" == "1" ]]; then
        # MCMCStrategy: stochastic relocation, no hard opacity resets, no runaway growth.
        # Preset "mcmc" already sets opacity_reg=0.01, scale_reg=0.01, init_opa=0.5, init_scale=0.1.
        # Scale refine_stop proportionally (default 25K/30K = 83%).
        _GS_MCMC_STOP=$(( ITERS * 5 / 6 ))
        _GSPLAT_VIEWER_ARGS=("--disable-viewer")
        [[ -n "$VIEWER_PORT" ]] && _GSPLAT_VIEWER_ARGS+=("--port" "$VIEWER_PORT")
        PYTHONPATH="$REPO/../gsplat/examples" TORCH_CUDA_ARCH_LIST="8.9" CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON" "$REPO/../gsplat/examples/simple_trainer.py" mcmc \
            --data-dir "$SCENE_DIR" \
            --data-factor "${GSPLAT_DATA_FACTOR:-1}" \
            --result-dir "$GSPLAT_OUT" \
            --max-steps "$ITERS" \
            --eval-steps "$ITERS" \
            --save-steps "$ITERS" \
            --ply-steps "$ITERS" \
            --save-ply \
            --init-type sfm \
            --fast-init \
            "${_GSPLAT_VIEWER_ARGS[@]}" \
            --disable-video \
            --ssim-lambda "$GSPLAT_SSIM_LAMBDA" \
            "${_GSPLAT_PP_ARGS[@]}" \
            --opacity-reg 0.05 \
            --strategy.cap-max "${GSPLAT_CAP_MAX:-2000000}" \
            --strategy.refine-stop-iter "$_GS_MCMC_STOP" \
            2>&1 | tee "$MODEL_DIR/02_train.log"
    else
        # DefaultStrategy + AbsGS. grow_grad2d MUST be raised from 0.0002 when using absgrad.
        # Scale refine_stop and reset_every proportionally for shorter runs.
        _GS_REFINE_STOP=$(( ITERS / 2 ))
        _GS_RESET_EVERY=$(( ITERS / 10 < 100 ? 100 : ITERS / 10 ))
        _GSPLAT_VIEWER_ARGS=("--disable-viewer")
        [[ -n "$VIEWER_PORT" ]] && _GSPLAT_VIEWER_ARGS+=("--port" "$VIEWER_PORT")
        PYTHONPATH="$REPO/../gsplat/examples" TORCH_CUDA_ARCH_LIST="8.9" CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/../gsplat/examples/simple_trainer.py" default \
            --data-dir "$SCENE_DIR" \
            --data-factor "${GSPLAT_DATA_FACTOR:-1}" \
            --result-dir "$GSPLAT_OUT" \
            --max-steps "$ITERS" \
            --eval-steps "$ITERS" \
            --save-steps "$ITERS" \
            --ply-steps "$ITERS" \
            --save-ply \
            --init-type sfm \
            --fast-init \
            "${_GSPLAT_VIEWER_ARGS[@]}" \
            --disable-video \
            --ssim-lambda "$GSPLAT_SSIM_LAMBDA" \
            "${_GSPLAT_PP_ARGS[@]}" \
            --strategy.refine-stop-iter "$_GS_REFINE_STOP" \
            --strategy.reset-every "$_GS_RESET_EVERY" \
            --strategy.absgrad \
            --strategy.grow-grad2d 0.0006 \
            --strategy.prune-opa 0.05 \
            --opacity-reg 0.01 \
            --scale-reg 0.01 \
            2>&1 | tee "$MODEL_DIR/02_train.log"
    fi
    # Extract initial camera (uses colmap_to_ply_transform.npy saved by simple_trainer.py)
    python3 "$REPO/camera_from_colmap.py" \
        --sparse     "$SPARSE_PARENT/0" \
        --trainer    gsplat \
        --result-dir "$GSPLAT_OUT" \
        --splat      "$(ls "$MODEL_DIR"/*.splat 2>/dev/null | head -1)" \
        --out        "$MODEL_DIR/initial_camera.json" \
        2>/dev/null || true
elif [[ "$TRAINER" == "2dgs" ]]; then
    [[ "$SFM" != "colmap_sift" && "$SFM" != "glomap_sift" && "$SFM" != "glomap_aliked" && "$SFM" != "glomap_loftr" && "$SFM" != "glomap_disk" && "$SFM" != "glomap_superpoint" && "$SFM" != "colmap_aliked" && "$SFM" != "fastmap" && "$SFM" != "realityscan" && "$SFM" != "preposed" ]] && ln -sfn "sparse_${TOTAL_FRAMES}" "$SCENE_DIR/sparse" 2>/dev/null || true
    GSPLAT_OUT="$MODEL_DIR/gsplat_output"
    _GS_REFINE_STOP=$(( ITERS / 2 ))
    echo "[2/3] 2DGS training ($ITERS iterations, $TOTAL_FRAMES frames)..."
    PYTHONPATH="$REPO/../gsplat/examples" TORCH_CUDA_ARCH_LIST="8.9" CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/../gsplat/examples/simple_trainer_2dgs.py" \
        --data-dir "$SCENE_DIR" \
        --data-factor 1 \
        --result-dir "$GSPLAT_OUT" \
        --max-steps "$ITERS" \
        --eval-steps "$ITERS" \
        --save-steps "$ITERS" \
        --ply-steps "$ITERS" \
        --save-ply \
        --init-type sfm \
        --fast-init \
        --disable-viewer \
        --disable-video \
        --ssim-lambda "$GSPLAT_SSIM_LAMBDA" \
        --prune-opa 0.05 \
        --refine-stop-iter "$_GS_REFINE_STOP" \
        2>&1 | tee "$MODEL_DIR/02_train.log"
    python3 "$REPO/camera_from_colmap.py" \
        --sparse     "$SPARSE_PARENT/0" \
        --trainer    gsplat \
        --result-dir "$GSPLAT_OUT" \
        --splat      "$(ls "$MODEL_DIR"/*.splat 2>/dev/null | head -1)" \
        --out        "$MODEL_DIR/initial_camera.json" \
        2>/dev/null || true
elif [[ "$TRAINER" == "brush" ]]; then
    # Brush: Rust-based MCMC-style trainer; headless by default (no --with-viewer).
    # Accepts COLMAP or nerfstudio format (auto-detected from scene_dir).
    [[ "$SFM" != "colmap_sift" && "$SFM" != "glomap_sift" && "$SFM" != "glomap_aliked" && \
       "$SFM" != "glomap_loftr" && "$SFM" != "glomap_disk" && "$SFM" != "glomap_superpoint" && \
       "$SFM" != "colmap_aliked" && "$SFM" != "fastmap" && "$SFM" != "realityscan" && \
       "$SFM" != "preposed" ]] && \
        ln -sfn "sparse_${TOTAL_FRAMES}" "$SCENE_DIR/sparse" 2>/dev/null || true
    BRUSH_BIN="${BRUSH_BIN:-/home/communications/workdir/brush/brush-app-x86_64-unknown-linux-gnu/brush_app}"
    BRUSH_OUT="$MODEL_DIR/brush_output"
    mkdir -p "$BRUSH_OUT"
    # For preposed: pass MODEL_DIR (has sparse/ but NOT transforms.json) so brush
    # uses its COLMAP loader instead of the nerfstudio loader (which ignores our
    # coordinate-corrected COLMAP sparse and loads the original PLY instead)
    _BRUSH_SCENE="$SCENE_DIR"
    [[ "$SFM" == "preposed" ]] && _BRUSH_SCENE="$MODEL_DIR"
    echo "[2/3] Brush training ($ITERS iterations, $TOTAL_FRAMES frames)..."
    RUST_LOG=brush_cli=info "$BRUSH_BIN" "$_BRUSH_SCENE" \
        --total-steps "$ITERS" \
        --export-path "$BRUSH_OUT" \
        --export-every "$ITERS" \
        --eval-split-every 8 \
        2>&1 | tee "$MODEL_DIR/02_train.log"
    SPARSE_PARENT="$SCENE_DIR/sparse"
    python3 "$REPO/camera_from_colmap.py" \
        --sparse     "$SPARSE_PARENT/0" \
        --trainer    brush \
        --result-dir "$BRUSH_OUT" \
        --splat      "$(ls "$MODEL_DIR"/*.splat 2>/dev/null | head -1)" \
        --out        "$MODEL_DIR/initial_camera.json" \
        2>/dev/null || true
else
    # instantsplat trainer
    # train.py looks for sparse_{N}/0/ — COLMAP/GLOMAP/FastMap write sparse/0/ instead;
    # symlink sparse_N → sparse so the scene loader finds it.
    [[ "$SFM" == "colmap_sift" || "$SFM" == "glomap_sift" || "$SFM" == "glomap_aliked" || "$SFM" == "glomap_loftr" || "$SFM" == "glomap_disk" || "$SFM" == "glomap_superpoint" || "$SFM" == "colmap_aliked" || "$SFM" == "fastmap" || "$SFM" == "realityscan" ]] && ln -sfn "sparse" "$SCENE_DIR/sparse_${TOTAL_FRAMES}" 2>/dev/null || true
    # --pp_optimizer requires confidence_dsp.npy from init_geo.py (MASt3R/Fast3R only)
    PP_OPT_ARG="--pp_optimizer"
    [[ "$SFM" == "colmap_sift" || "$SFM" == "glomap_sift" || "$SFM" == "glomap_aliked" || "$SFM" == "glomap_loftr" || "$SFM" == "glomap_disk" || "$SFM" == "glomap_superpoint" || "$SFM" == "colmap_aliked" || "$SFM" == "fastmap" || "$SFM" == "realityscan" ]] && PP_OPT_ARG=""
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
    # instantsplat trainer: no colmap_to_ply_transform.npy yet; use sparse/0 directly
    # (output will be in COLMAP world space — still better than nothing)
    python3 "$REPO/camera_from_colmap.py" \
        --sparse     "$SPARSE_PARENT/0" \
        --trainer    instantsplat \
        --result-dir "$MODEL_DIR" \
        --splat      "$(ls "$MODEL_DIR"/*.splat 2>/dev/null | head -1)" \
        --out        "$MODEL_DIR/initial_camera.json" \
        2>/dev/null || true
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
METRICS_JSON=$("$PYTHON" "$REPO/extract_metrics.py" "$MODEL_DIR" "$SPARSE_PARENT/0" 2>/dev/null || echo "{}")
emit_event "{\"event\":\"train_done\",\"elapsed_s\":$TRAIN_ELAPSED,\"early_stopped\":$EARLY_STOPPED_FLAG,\"stopped_at_iter\":$STOPPED_AT_ITER,\"metrics\":$METRICS_JSON}"

# Resolve PLY path — differs by trainer
if [[ "$SFM" == "onthefly" ]]; then
    ONTHEFLY_OUT="$MODEL_DIR/onthefly_out"
    N_ANCHORS=$(find "$ONTHEFLY_OUT/point_clouds" -name "anchor_*.ply" 2>/dev/null | wc -l)
    (( N_ANCHORS > 1 )) && echo "    ⚠  on-the-fly produced $N_ANCHORS anchors — using anchor_0.ply (multi-anchor merge not implemented)"
    PLY="$ONTHEFLY_OUT/point_clouds/anchor_0.ply"
    [[ ! -f "$PLY" ]] && { echo "Error: $PLY not found. Check $MODEL_DIR/02_onthefly.log"; exit 1; }
    ACTUAL_ITER=$ONTHEFLY_ITERS
elif [[ "$TRAINER" == "gsplat" ]]; then
    PLY=$(find "$MODEL_DIR/gsplat_output/ply" -name "*.ply" 2>/dev/null | sort | tail -1)
    ACTUAL_ITER=$ITERS
elif [[ "$TRAINER" == "2dgs" ]]; then
    PLY=$(find "$MODEL_DIR/gsplat_output/point_cloud" -name "*.ply" 2>/dev/null | sort | tail -1)
    ACTUAL_ITER=$ITERS
elif [[ "$TRAINER" == "brush" ]]; then
    PLY=$(find "$MODEL_DIR/brush_output" -name "*.ply" 2>/dev/null | sort | tail -1)
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
# Remove sparse from assets/examples — canonical copy is now in the pod
if [[ "$SFM" == "colmap_sift" || "$SFM" == "glomap_sift" || "$SFM" == "glomap_aliked" || \
      "$SFM" == "glomap_loftr" || "$SFM" == "glomap_disk" || "$SFM" == "glomap_superpoint" || \
      "$SFM" == "colmap_aliked" || "$SFM" == "fastmap" || "$SFM" == "realityscan" ]]; then
    [[ -d "$SPARSE_PARENT" ]] && rm -rf "$SPARSE_PARENT"
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done! $(date '+%Y-%m-%d %H:%M:%S')  (${ELAPSED_MIN} min)"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  $SPLAT_OUT"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Drag onto https://antimatter15.com/splat/ to view"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
