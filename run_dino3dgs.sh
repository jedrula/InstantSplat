#!/bin/bash
# run_dino3dgs.sh — DINOv2 feature distillation into an existing 3DGS pod
#
# Usage:
#   bash run_dino3dgs.sh --ply /path/to/point_cloud.ply \
#                        --scene-dir /path/to/scene/ \
#                        --output-dir /path/to/dino_out/ \
#                        [--iters 3000] [--n-components 8]
#
# After pipeline, query with:
#   PYTHONPATH=/workdir/gsplat python InstantSplat/dino3dgs/query.py \
#     --dino-ply OUTPUT_DIR/dino_gaussians.ply \
#     --pca OUTPUT_DIR/features/pca.npz \
#     --scene-dir SCENE_DIR \
#     --mask mask.png \
#     --output heatmap.png

set -eo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${INSTANTSPLAT_PYTHON:-${HOME}/miniconda3/envs/instantsplat/bin/python}"

PLY=""
SCENE_DIR=""
OUTPUT_DIR=""
IMAGE_DIR=""
ITERS=3000
N_COMPONENTS=8
MAX_SIDE=560
LR=5e-3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ply)           PLY="$2";           shift 2 ;;
        --scene-dir)     SCENE_DIR="$2";     shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";    shift 2 ;;
        --image-dir)     IMAGE_DIR="$2";     shift 2 ;;
        --iters)         ITERS="$2";         shift 2 ;;
        --n-components)  N_COMPONENTS="$2";  shift 2 ;;
        --max-side)      MAX_SIDE="$2";      shift 2 ;;
        --lr)            LR="$2";            shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

[[ -z "$PLY" ]]        && { echo "Error: --ply required";        exit 1; }
[[ -z "$SCENE_DIR" ]]  && { echo "Error: --scene-dir required";  exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "Error: --output-dir required"; exit 1; }
[[ ! -f "$PLY" ]]      && { echo "Error: PLY not found: $PLY";   exit 1; }

[[ -z "$IMAGE_DIR" ]] && IMAGE_DIR="$SCENE_DIR/images"
[[ ! -d "$IMAGE_DIR" ]] && { echo "Error: image dir not found: $IMAGE_DIR"; exit 1; }

mkdir -p "$OUTPUT_DIR"
FEAT_DIR="$OUTPUT_DIR/features"
DINO_PLY="$OUTPUT_DIR/dino_gaussians.ply"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  DINOv2 → 3DGS Distillation"
echo "║  PLY:     $PLY"
echo "║  images:  $IMAGE_DIR"
echo "║  output:  $OUTPUT_DIR"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Phase 1: DINOv2 feature extraction + PCA"
echo "║  Phase 2: Gaussian distillation ($ITERS iters)"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

T0=$(date +%s)

# ── Phase 1: DINOv2 extraction + PCA ─────────────────────────────────────────
echo "[1/2] DINOv2 feature extraction (no SAM2, no segmentation)..."
mkdir -p "$FEAT_DIR"

"$PYTHON" "$REPO/dino3dgs/extract_features.py" \
    --image-dir    "$IMAGE_DIR" \
    --output-dir   "$FEAT_DIR" \
    --n-components "$N_COMPONENTS" \
    --max-side     "$MAX_SIDE" \
    2>&1 | tee "$OUTPUT_DIR/01_extract.log"

T1=$(date +%s)
echo "    → Phase 1: $(( T1 - T0 ))s"

# ── Phase 2: Distillation ─────────────────────────────────────────────────────
echo ""
echo "[2/2] Distilling DINO features into Gaussians ($ITERS iters)..."

PYTHONPATH="$REPO/../gsplat" "$PYTHON" "$REPO/dino3dgs/distill.py" \
    --ply        "$PLY" \
    --dino-dir   "$FEAT_DIR" \
    --scene-dir  "$SCENE_DIR" \
    --output-ply "$DINO_PLY" \
    --iters      "$ITERS" \
    --lr         "$LR" \
    2>&1 | tee "$OUTPUT_DIR/02_distill.log"

T2=$(date +%s)
TOTAL=$(( T2 - T0 ))

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Done! ($(awk "BEGIN{printf \"%.1f\",$TOTAL/60}") min total)"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Phase 1 (DINOv2 + PCA): $(( T1 - T0 ))s"
echo "║  Phase 2 (distillation):  $(( T2 - T1 ))s"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  DINO PLY: $DINO_PLY"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Query (mask mode):"
echo "║    PYTHONPATH=/workdir/gsplat \\"
echo "║    python $REPO/dino3dgs/query.py \\"
echo "║      --dino-ply $DINO_PLY \\"
echo "║      --pca $FEAT_DIR/pca.npz \\"
echo "║      --scene-dir $SCENE_DIR \\"
echo "║      --mask /path/to/mask.png \\"
echo "║      --output heatmap.png"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
