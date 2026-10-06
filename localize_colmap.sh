#!/usr/bin/env bash
# localize_colmap.sh — Localize a query image into an existing pod reconstruction.
#
# Usage:
#   bash localize_colmap.sh <pod_dir> <query_image> [output.json]
#
# <pod_dir>    : topowall-splat pod directory (must contain database.db and sparse/0/)
# <query_image>: path to the query image (any format readable by COLMAP)
# [output.json]: where to write initial_camera.json (default: <pod_dir>/localized_camera.json)
#
# Requires database.db in pod_dir — only present for jobs run after the
# "database preservation" fix (2026-06-16). Re-run the job to get it.

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${INSTANTSPLAT_PYTHON:-${HOME}/miniconda3/envs/instantsplat/bin/python}"   # needs pycolmap

POD_DIR="${1:?Usage: $0 <pod_dir> <query_image> [output.json] [--vocab-tree]}"
QUERY_IMG="${2:?Usage: $0 <pod_dir> <query_image> [output.json] [--vocab-tree]}"
OUTPUT_JSON="${3:-$POD_DIR/localized_camera.json}"
USE_VOCAB_TREE="${4:-}"  # pass --vocab-tree to use vocab_tree_matcher (slower for small scenes)

# ── COLMAP ───────────────────────────────────────────────────────────────────
# Use the binary that BUILT this pod, which the pod records itself. Query descriptors must come
# from it: GPU-SIFT format changed between COLMAP generations and a mismatch yields 0 verified
# pairs with no error, which reads like a bad query image.
#
# No version dispatch and no default path here on purpose — the pod is the single source of
# truth, so there is nothing to keep in sync with video_to_splat.sh. Pods built before
# 2026-09-07 have no marker and must have their SfM re-run to be localizable.
MARKER="$POD_DIR/colmap_version.txt"
[[ -f "$MARKER" ]] || { echo "ERROR: $MARKER missing — pod predates COLMAP version recording (2026-09-07). Re-run its SfM." >&2; exit 1; }
COLMAP=$(head -1 "$MARKER")
[[ -x "$COLMAP" ]] || { echo "ERROR: this pod was built with '$COLMAP', which is not executable here." >&2; exit 1; }
# CPU and GPU SIFT do not produce interchangeable descriptors, so the query must be extracted
# the same way the pod was — same class of silent "0 verified pairs" failure as a version
# mismatch. Markers written before --sfm-sift-gpu existed carry no such line and are GPU-built.
POD_SIFT_GPU=$(sed -n 's/^sift_gpu=//p' "$MARKER")
POD_SIFT_GPU="${POD_SIFT_GPU:-1}"
GPU_FLAG_EXTRACT="--FeatureExtraction.use_gpu $POD_SIFT_GPU"
GPU_FLAG_MATCH="--FeatureMatching.use_gpu 1"
echo "  COLMAP: $(sed -n 2p "$MARKER")  (SIFT: $([[ "$POD_SIFT_GPU" == 1 ]] && echo GPU || echo CPU))"

VOCAB_TREE="$REPO/assets/vocab_tree_flickr100K_words32K.bin"

SPARSE_IN="$POD_DIR/sparse/0"
DATABASE="$POD_DIR/database.db"

# ── Validate inputs ───────────────────────────────────────────────────────────
[[ -f "$DATABASE" ]]           || { echo "ERROR: $DATABASE not found. Re-run the job to generate it."; exit 1; }
[[ -f "$QUERY_IMG" ]]          || { echo "ERROR: query image not found: $QUERY_IMG"; exit 1; }
[[ -d "$SPARSE_IN" ]]          || { echo "ERROR: sparse/0 not found in $POD_DIR"; exit 1; }
[[ -f "$SPARSE_IN/images.bin" ]] || { echo "ERROR: no images.bin in $SPARSE_IN"; exit 1; }
if [[ "$USE_VOCAB_TREE" == "--vocab-tree" ]]; then
    [[ -f "$VOCAB_TREE" ]] || { echo "ERROR: vocab tree not found: $VOCAB_TREE"; exit 1; }
fi

export QT_QPA_PLATFORM=offscreen

# ── Temp workspace ────────────────────────────────────────────────────────────
TMPDIR=$(mktemp -d /tmp/localize_XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

# Remap database image IDs to match the sparse model (GLOMAP reassigns them)
DB_RAW="$TMPDIR/database_raw.db"
DB="$TMPDIR/database.db"
cp "$DATABASE" "$DB_RAW"
"$PYTHON" "$REPO/remap_db_to_sparse.py" "$DB_RAW" "$SPARSE_IN" "$DB"

# Query image goes in its own directory so feature_extractor only touches it
QUERY_DIR="$TMPDIR/query"
mkdir -p "$QUERY_DIR"
QUERY_NAME="$(basename "$QUERY_IMG")"
# COLMAP needs the image inside the image_path directory
ln -s "$(realpath "$QUERY_IMG")" "$QUERY_DIR/$QUERY_NAME"

SPARSE_OUT="$TMPDIR/sparse"
mkdir -p "$SPARSE_OUT"

echo "=== localize_colmap ==="
echo "  pod:   $POD_DIR"
echo "  query: $QUERY_IMG"
echo "  db:    $DB (copy)"

# ── 1. Extract features for query image ──────────────────────────────────────
echo ""
echo "--- Step 1: Feature extraction ---"
"$COLMAP" feature_extractor \
    --database_path "$DB" \
    --image_path "$QUERY_DIR" \
    $GPU_FLAG_EXTRACT \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 0 \
    2>&1 | tail -5

# ── 2. Match query against training images ────────────────────────────────────
echo ""
if [[ "$USE_VOCAB_TREE" == "--vocab-tree" ]]; then
    # Vocab tree: better for large scenes (>500 images), slower for small ones
    echo "--- Step 2: Vocab tree matching ---"
    "$COLMAP" vocab_tree_matcher \
        --database_path "$DB" \
        --VocabTreeMatching.vocab_tree_path "$VOCAB_TREE" \
        $GPU_FLAG_MATCH \
        2>&1 | tail -10
else
    # Exhaustive: faster than vocab tree for scenes ≤500 images.
    # Skips pairs already in matches table, so only runs the 1×N new query pairs.
    echo "--- Step 2: Exhaustive matching ---"
    "$COLMAP" exhaustive_matcher \
        --database_path "$DB" \
        $GPU_FLAG_MATCH \
        2>&1 | tail -5
fi

# ── 3. Register query into existing reconstruction ────────────────────────────
echo ""
echo "--- Step 3: Image registration ---"
"$COLMAP" image_registrator \
    --database_path "$DB" \
    --input_path "$SPARSE_IN" \
    --output_path "$SPARSE_OUT" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.min_focal_length_ratio 0.1 \
    --Mapper.max_focal_length_ratio 10 \
    2>&1 | tail -15

# ── 4. Extract pose from output model ─────────────────────────────────────────
echo ""
echo "--- Step 4: Extracting pose ---"

REGISTERED_SPARSE=$(find "$SPARSE_OUT" -name "images.bin" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)

if [[ -z "$REGISTERED_SPARSE" ]]; then
    echo "ERROR: image_registrator produced no output — query image could not be registered."
    echo "  Possible causes: too few feature matches, or query has no overlap with reconstruction."
    exit 1
fi

"$PYTHON" - <<PYEOF
import json, math, sys
from pathlib import Path
import pycolmap

rec          = pycolmap.Reconstruction("$REGISTERED_SPARSE")
query_name   = "$QUERY_NAME"
output_json  = Path("$OUTPUT_JSON")

query = next((im for im in rec.images.values() if Path(im.name).name == query_name), None)
if query is None or not query.has_pose:
    print(f"WARNING: '{query_name}' not found in registered model.", file=sys.stderr)
    print(f"  Registered images: {[Path(im.name).name for im in rec.images.values()]}", file=sys.stderr)
    sys.exit(1)

# 2D-3D inlier correspondences = the query's observations that carry a 3D point
obs_ids = [p2d.point3D_id for p2d in query.points2D if p2d.has_point3D()]
inliers = len(obs_ids)
visible_ids = set(obs_ids)

qx, qy, qz, qw = query.cam_from_world().rotation.quat   # pycolmap stores xyzw
tx, ty, tz     = query.cam_from_world().translation

# Normalise quaternion
n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n

# world-to-camera rotation matrix
R = [
    [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
    [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
    [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
]
t = [tx, ty, tz]

# Camera world position: -R^T @ t
pos = [-sum(R[j][k]*t[j] for j in range(3)) for k in range(3)]

# Forward in world: R^T @ [0,0,1]  =  row 2 of R_w2c  (NOT col 2)
fwd = [R[2][j] for j in range(3)]
n_fwd = math.sqrt(sum(x*x for x in fwd))
fwd = [x/n_fwd for x in fwd]

# Up in world: -R^T @ [0,1,0]  =  -(row 1 of R_w2c)  (NOT -col 1)
up = [-R[1][j] for j in range(3)]
n_up = math.sqrt(sum(x*x for x in up))
up = [x/n_up for x in up]

# look_at: use pts3D centroid directly (same approach as camera_from_colmap.py).
# No projection onto fwd — works regardless of camera orientation.
pts3d_by_id = {pid: list(p.xyz) for pid, p in rec.points3D.items()}

pts3d = list(pts3d_by_id.values())
if not pts3d:
    sys.exit("ERROR: registered sparse has no 3D points — cannot compute look_at")

# Use centroid of 3D points visible from the query image so look_at points
# at the section of the wall actually in the frame, not the whole scene center.
visible_pts = [pts3d_by_id[pid] for pid in visible_ids if pid in pts3d_by_id]
target_pts = visible_pts if visible_pts else pts3d

cx = sum(p[0] for p in target_pts) / len(target_pts)
cy = sum(p[1] for p in target_pts) / len(target_pts)
cz = sum(p[2] for p in target_pts) / len(target_pts)
look_at = [cx, cy, cz]

to_c = [cx - pos[0], cy - pos[1], cz - pos[2]]
look_dist = sum(to_c[i]*fwd[i] for i in range(3))

result = {
    "position":     [round(v, 4) for v in pos],
    "look_at":      [round(v, 4) for v in look_at],
    "up":           [round(v, 4) for v in up],
    "source_frame": f"localized:{query_name}",
    "debug": {
        "inliers":    inliers,
        "n_pts3d":    len(pts3d),
        "look_dist":  round(look_dist, 4),
        "fwd":        [round(v, 4) for v in fwd],
    },
}
output_json.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print(f"  inliers={inliers}  pts3d={len(pts3d)}  look_dist={look_dist:.3f}", file=sys.stderr)
PYEOF

echo ""
echo "Wrote $OUTPUT_JSON"
