#!/bin/bash
# Generate and evaluate meshes for all Replica scenes using cameras.json method
# This script uses the --gt_traj flag to align mesh to GT coordinate system

set -e

RESULT_BASE=$1  # e.g., results_pa11
VOXEL_SIZE=${2:-0.01}
DEPTH_SCALE=${3:-6553.5}

REPLICA_BASE="/media/tam/DATA/data/Replica"
GT_MESH_BASE="/media/tam/DATA/data/Replica/cull_replica_mesh"
TSDF_ENV="/media/tam/DATA/3D/CG-photo/scripts/tsdf_env/bin/activate"

SCENES=("office0" "office1" "office2" "office3" "office4" "room0" "room1" "room2")

echo "=============================================="
echo "Generating meshes from cameras.json"
echo "Result base: $RESULT_BASE"
echo "Voxel size: $VOXEL_SIZE"
echo "Depth scale: $DEPTH_SCALE"
echo "=============================================="

source $TSDF_ENV

for scene in "${SCENES[@]}"; do
    echo ""
    echo "=== Processing $scene ==="
    
    # Find the shutdown directory (contains ply/cameras.json)
    RESULT_DIR=$(find "$RESULT_BASE/${scene}_run1" -name "*_shutdown" -type d 2>/dev/null | head -1)
    
    if [ -z "$RESULT_DIR" ]; then
        echo "Warning: No shutdown directory found for $scene in $RESULT_BASE/${scene}_run1"
        continue
    fi
    
    JSON_PATH="$RESULT_DIR/ply/cameras.json"
    DEPTH_DIR="$RESULT_DIR/depth"
    GT_TRAJ="$REPLICA_BASE/$scene/traj.txt"
    OUTPUT_DIR="$RESULT_BASE/${scene}_run1/meshes"
    OUTPUT_MESH="$OUTPUT_DIR/${scene}_json_aligned.ply"
    GT_MESH="$GT_MESH_BASE/${scene}.ply"
    
    if [ ! -f "$JSON_PATH" ]; then
        echo "Warning: cameras.json not found: $JSON_PATH"
        continue
    fi
    
    if [ ! -f "$GT_TRAJ" ]; then
        echo "Warning: GT trajectory not found: $GT_TRAJ"
        continue
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    echo "Generating mesh for $scene..."
    python scripts/generate_mesh_from_json.py \
        --json_path "$JSON_PATH" \
        --depth_dir "$DEPTH_DIR" \
        --output "$OUTPUT_MESH" \
        --voxel_size $VOXEL_SIZE \
        --depth_scale $DEPTH_SCALE \
        --max_depth 10.0 \
        --gt_traj "$GT_TRAJ"
    
    echo "Evaluating mesh for $scene..."
    if [ -f "$OUTPUT_MESH" ] && [ -f "$GT_MESH" ]; then
        python neural_slam_eval-main/eval_recon.py \
            --rec_mesh "$OUTPUT_MESH" \
            --gt_mesh "$GT_MESH" \
            --dataset_type Replica \
            -3d
    else
        echo "Warning: Cannot evaluate - mesh or GT not found"
    fi
    
    echo "=== Done $scene ==="
done

echo ""
echo "=============================================="
echo "All scenes processed!"
echo "=============================================="
