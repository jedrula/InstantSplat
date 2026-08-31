#!/usr/bin/env bash
# DINOv2 distillation wrapper for 3DGS scenes.
#
# Runs the three-stage pipeline:
#   1. extract_features.py  — DINOv2 → PCA-compressed feature maps
#   2. distill.py           — train per-Gaussian DINO features (geometry frozen)
#   3. (query.py runs separately per query)
#
# Usage:
#   bash dino3dgs/run_dino_distill.sh \
#       --scene  fpinka-full-sparse \
#       [--iters 3000] \
#       [--n-components 8] \
#       [--ply /explicit/path/to/point_cloud.ply] \
#       [--model-dir /explicit/model/dir]
#
# By default:
#   IMAGE_DIR  = $REPO/assets/examples/$SCENE/images
#   SCENE_DIR  = $REPO/assets/examples/$SCENE
#   MODEL_DIR  = $REPO/output_infer/$SCENE
#   PLY        = newest point_cloud.ply under MODEL_DIR/point_cloud/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINO_DIR="$REPO/dino3dgs"
PYTHON="${INSTANTSPLAT_PYTHON:-$(conda run -n instantsplat which python 2>/dev/null || echo python3)}"

SCENE=""
ITERS=3000
N_COMPONENTS=8
PLY_OVERRIDE=""
MODEL_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene)        SCENE="$2"; shift 2 ;;
        --iters)        ITERS="$2"; shift 2 ;;
        --n-components) N_COMPONENTS="$2"; shift 2 ;;
        --ply)          PLY_OVERRIDE="$2"; shift 2 ;;
        --model-dir)    MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$SCENE" && -z "$PLY_OVERRIDE" ]]; then
    echo "Error: --scene SCENE_NAME or --ply /path/to/point_cloud.ply required"
    exit 1
fi

# Resolve paths
if [[ -n "$MODEL_DIR_OVERRIDE" ]]; then
    MODEL_DIR="$MODEL_DIR_OVERRIDE"
elif [[ -n "$SCENE" ]]; then
    MODEL_DIR="$REPO/output_infer/$SCENE"
fi

SCENE_DIR="$REPO/assets/examples/$SCENE"
IMAGE_DIR="$SCENE_DIR/images"

# Validate image dir
if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "Error: image dir not found: $IMAGE_DIR"
    exit 1
fi

# Find PLY
if [[ -n "$PLY_OVERRIDE" ]]; then
    PLY="$PLY_OVERRIDE"
else
    PLY=$(find "$MODEL_DIR/point_cloud" -name "point_cloud.ply" \
              | sort -t_ -k2 -rn | head -1)
    if [[ -z "$PLY" ]]; then
        echo "Error: no point_cloud.ply under $MODEL_DIR/point_cloud/"
        exit 1
    fi
fi
echo "[dino] Using PLY: $PLY"
echo "[dino] Images:    $IMAGE_DIR"
echo "[dino] Scene dir: $SCENE_DIR"

DINO_OUT="$MODEL_DIR/dino"
mkdir -p "$DINO_OUT"

# ── Stage 1: extract DINOv2 features ─────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Stage 1/2: DINOv2 feature extraction"
echo "══════════════════════════════════════════"
PYTHONPATH="$REPO/gsplat" "$PYTHON" "$DINO_DIR/extract_features.py" \
    --image-dir   "$IMAGE_DIR" \
    --output-dir  "$DINO_OUT" \
    --n-components "$N_COMPONENTS" \
    2>&1 | tee "$MODEL_DIR/dino_extract.log"

# ── Stage 2: distill into Gaussians ──────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " Stage 2/2: Distill into Gaussians"
echo "══════════════════════════════════════════"
OUTPUT_PLY="$DINO_OUT/dino_gaussians.ply"
PYTHONPATH="$REPO/gsplat" "$PYTHON" "$DINO_DIR/distill.py" \
    --ply        "$PLY" \
    --dino-dir   "$DINO_OUT" \
    --scene-dir  "$SCENE_DIR" \
    --output-ply "$OUTPUT_PLY" \
    --iters      "$ITERS" \
    2>&1 | tee "$MODEL_DIR/dino_distill.log"

echo ""
echo "══════════════════════════════════════════"
echo " Done!"
echo "  DINO PLY:  $OUTPUT_PLY"
echo "  PCA:       $DINO_OUT/pca.npz"
echo ""
echo " To query:"
echo "  PYTHONPATH=$REPO/gsplat python $DINO_DIR/query.py \\"
echo "    --dino-ply $OUTPUT_PLY \\"
echo "    --pca $DINO_OUT/pca.npz \\"
echo "    --scene-dir $SCENE_DIR \\"
echo "    --mask my_mask.png \\"
echo "    --output heatmap.png"
echo "══════════════════════════════════════════"
