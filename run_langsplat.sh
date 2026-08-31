#!/bin/bash
# run_langsplat.sh — Full LangSplat pipeline on an existing 3DGS pod
#
# Usage:
#   bash run_langsplat.sh --ply /path/to/point_cloud.ply \
#                         --scene-dir /path/to/scene_dir \
#                         --output-dir /path/to/langsplat_output \
#                         [--iters 3000] [--ae-epochs 30]
#
# Phases:
#   1. Feature extraction (SAM2 + CLIP)  — runs with sam2 venv python
#   2. Autoencoder training              — runs with instantsplat python
#   3. Language Gaussian training        — runs with instantsplat python
#
# Query (after pipeline):
#   PYTHONPATH=/workdir/gsplat \
#   python InstantSplat/langsplat/query.py \
#     --lang-ply $OUTPUT_DIR/lang_gaussians.ply \
#     --autoencoder $OUTPUT_DIR/ae/autoencoder.pth \
#     --query "red hold" \
#     --camera-json $POD_DIR/initial_camera.json \
#     --output query_heatmap.png

set -eo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${INSTANTSPLAT_PYTHON:-${HOME}/miniconda3/envs/instantsplat/bin/python}"
SAM2_PYTHON="/home/communications/workdir/sam2/venv/bin/python"

# ── Arg parsing ──────────────────────────────────────────────────────────────
PLY=""
SCENE_DIR=""
OUTPUT_DIR=""
IMAGE_DIR=""
ITERS=3000
AE_EPOCHS=30
LATENT_DIM=3
AE_LR=1e-3
LANG_LR=5e-3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ply)        PLY="$2";        shift 2 ;;
        --scene-dir)  SCENE_DIR="$2";  shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --image-dir)  IMAGE_DIR="$2";  shift 2 ;;
        --iters)      ITERS="$2";      shift 2 ;;
        --ae-epochs)  AE_EPOCHS="$2";  shift 2 ;;
        --latent-dim) LATENT_DIM="$2"; shift 2 ;;
        --ae-lr)      AE_LR="$2";      shift 2 ;;
        --lang-lr)    LANG_LR="$2";    shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$PLY" ]]        && { echo "Error: --ply required"; exit 1; }
[[ -z "$SCENE_DIR" ]]  && { echo "Error: --scene-dir required"; exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "Error: --output-dir required"; exit 1; }
[[ ! -f "$PLY" ]]      && { echo "Error: PLY not found: $PLY"; exit 1; }
[[ ! -d "$SCENE_DIR" ]] && { echo "Error: scene dir not found: $SCENE_DIR"; exit 1; }

# Default image dir: $SCENE_DIR/images
[[ -z "$IMAGE_DIR" ]] && IMAGE_DIR="$SCENE_DIR/images"
[[ ! -d "$IMAGE_DIR" ]] && { echo "Error: image dir not found: $IMAGE_DIR (use --image-dir)"; exit 1; }

mkdir -p "$OUTPUT_DIR"

FEAT_DIR="$OUTPUT_DIR/features"
AE_DIR="$OUTPUT_DIR/ae"
LANG_PLY="$OUTPUT_DIR/lang_gaussians.ply"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  LangSplat Pipeline"
echo "║  PLY:      $PLY"
echo "║  scene:    $SCENE_DIR"
echo "║  images:   $IMAGE_DIR"
echo "║  output:   $OUTPUT_DIR"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Phase 1: Feature extraction (SAM2 + CLIP)"
echo "║  Phase 2: Autoencoder training"
echo "║  Phase 3: Language Gaussian training ($ITERS iters)"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Phase 1: Feature Extraction ──────────────────────────────────────────────
START=$(date +%s)
echo "[1/3] Feature extraction (SAM2 + CLIP)..."
mkdir -p "$FEAT_DIR"

if [[ ! -f "$FEAT_DIR/features_meta.json" ]]; then
    "$SAM2_PYTHON" "$REPO/langsplat/extract_features.py" \
        --image-dir "$IMAGE_DIR" \
        --output-dir "$FEAT_DIR" \
        --scale 4 \
        2>&1 | tee "$OUTPUT_DIR/01_extract_features.log"
else
    echo "    → Skipping (features_meta.json already exists)"
fi
P1=$(date +%s)
echo "    → Phase 1 done in $(( P1 - START ))s"

# ── Phase 2: Autoencoder ─────────────────────────────────────────────────────
echo ""
echo "[2/3] Autoencoder training (${AE_EPOCHS} epochs)..."
mkdir -p "$AE_DIR"

if [[ ! -f "$AE_DIR/autoencoder.pth" ]]; then
    "$PYTHON" "$REPO/langsplat/train_autoencoder.py" \
        --feature-dir "$FEAT_DIR" \
        --output-dir  "$AE_DIR" \
        --epochs      "$AE_EPOCHS" \
        --lr          "$AE_LR" \
        --latent-dim  "$LATENT_DIM" \
        2>&1 | tee "$OUTPUT_DIR/02_train_autoencoder.log"
else
    echo "    → Skipping (autoencoder.pth already exists)"
fi
P2=$(date +%s)
echo "    → Phase 2 done in $(( P2 - P1 ))s"

# ── Phase 3: Language Gaussian Training ──────────────────────────────────────
echo ""
echo "[3/3] Language Gaussian training (${ITERS} iters)..."

PYTHONPATH="$REPO/../gsplat" "$PYTHON" "$REPO/langsplat/train_lang_gaussians.py" \
    --ply        "$PLY" \
    --lang-dir   "$AE_DIR" \
    --scene-dir  "$SCENE_DIR" \
    --output-ply "$LANG_PLY" \
    --iters      "$ITERS" \
    --lr         "$LANG_LR" \
    --latent-dim "$LATENT_DIM" \
    2>&1 | tee "$OUTPUT_DIR/03_train_lang_gaussians.log"

P3=$(date +%s)
TOTAL=$(( P3 - START ))
TOTAL_MIN=$(awk "BEGIN {printf \"%.1f\", $TOTAL / 60}")

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  LangSplat Done! (${TOTAL_MIN} min total)"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Lang PLY:      $LANG_PLY"
echo "║  Autoencoder:   $AE_DIR/autoencoder.pth"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Query example:"
echo "║    PYTHONPATH=/workdir/gsplat \\"
echo "║    python $REPO/langsplat/query.py \\"
echo "║      --lang-ply $LANG_PLY \\"
echo "║      --autoencoder $AE_DIR/autoencoder.pth \\"
echo "║      --query \"red climbing hold\" \\"
echo "║      --scene-dir $SCENE_DIR \\"
echo "║      --output query_heatmap.png"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
