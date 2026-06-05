#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# test_splat_pipeline.sh — tight-loop test for Gaussian Splat training
#
# Reuses a saved Fast3R output dir (with .npy arrays) and re-runs just the
# COLMAP prep + training + ply2splat steps, skipping the Gradio UI entirely.
#
# Usage:
#   bash test_splat_pipeline.sh [work_dir] [iterations]
#
# Defaults:
#   work_dir   = most recently modified output dir under fast3r/demo_outputs
#   iterations = 500
# ---------------------------------------------------------------------------
set -e

INSTANTSPLAT_DIR="/home/communications/workdir/InstantSplat"
PYTHON="/home/communications/miniconda3/envs/instantsplat/bin/python"

# ── resolve work_dir ────────────────────────────────────────────────────────
if [[ -n "$1" ]]; then
    WORK_DIR="$(realpath "$1")"
else
    WORK_DIR=$(find /home/communications/workdir/fast3r/demo_outputs -name "pts3d.npy" -printf '%T@ %h\n' 2>/dev/null \
               | sort -n | tail -1 | awk '{print $2}')
    if [[ -z "$WORK_DIR" ]]; then
        echo "ERROR: no saved Fast3R output found. Run inference at least once via Gradio first." >&2
        exit 1
    fi
fi

ITERATIONS="${2:-500}"
N_VIEWS=$(cat "$WORK_DIR/n_views.txt")
SOURCE_PATH="$WORK_DIR/splat_source"
MODEL_PATH="$WORK_DIR/splat_model"

echo "============================================================"
echo "  work_dir    : $WORK_DIR"
echo "  n_views     : $N_VIEWS"
echo "  iterations  : $ITERATIONS"
echo "  source_path : $SOURCE_PATH"
echo "  model_path  : $MODEL_PATH"
echo "============================================================"

# ── 1. COLMAP prep ──────────────────────────────────────────────────────────
echo ""
echo "── Step 1: COLMAP prep ────────────────────────────────────"
cd "$INSTANTSPLAT_DIR"
"$PYTHON" prep_colmap_from_arrays.py "$WORK_DIR"

# ── 2. Train ────────────────────────────────────────────────────────────────
echo ""
echo "── Step 2: 3DGS training ($ITERATIONS iters) ──────────────"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" train.py \
    -s "$SOURCE_PATH" \
    -m "$MODEL_PATH" \
    --n_views "$N_VIEWS" \
    --iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --sh_degree 0 \
    --densify_grad_threshold 0.0005 \
    --disable_viewer

# ── 3. ply → splat ──────────────────────────────────────────────────────────
echo ""
echo "── Step 3: ply → .splat ───────────────────────────────────"
PLY_PATH="$MODEL_PATH/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
SPLAT_PATH="$MODEL_PATH/output.splat"

"$PYTHON" ply2splat.py "$PLY_PATH" "$SPLAT_PATH"

echo ""
echo "============================================================"
echo "  DONE — splat file: $SPLAT_PATH"
echo "  View with:"
echo "    bash $INSTANTSPLAT_DIR/view_splat.sh $SPLAT_PATH"
echo "============================================================"
