#!/bin/bash
# Evaluate all metrics for a Photo-SLAM result directory
# Usage: ./eval_all_metrics.sh <result_dir> [scene_name]
# Example: ./eval_all_metrics.sh results_pa9/office0 office0

set -e

RESULT_DIR="${1:?Missing result directory}"
SCENE_NAME="${2:-$(basename $RESULT_DIR)}"

# Paths
BASE_DIR="/media/tam/DATA/3D/CG-photo"
GT_DATA_DIR="/media/tam/DATA/data/Replica/${SCENE_NAME}"
GT_MESH="/media/tam/DATA/data/Replica/cull_replica_mesh/${SCENE_NAME}.ply"
VENV_PATH="/media/tam/DATA/3D/Photo-SLAM/venv"
TSDF_ENV_PATH="${BASE_DIR}/scripts/tsdf_env"

cd "$BASE_DIR"

echo "=============================================="
echo "  Evaluating: ${RESULT_DIR}"
echo "  Scene: ${SCENE_NAME}"
echo "=============================================="
echo ""

# 1. Photometric Metrics
echo "=== PHOTOMETRIC METRICS ==="
if [ -f "${RESULT_DIR}/psnr.txt" ]; then
    PSNR=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/psnr.txt")
    SSIM=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/ssim.txt")
    LPIPS=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/lpips.txt")
    echo "PSNR:  ${PSNR} dB"
    echo "SSIM:  ${SSIM}"
    echo "LPIPS: ${LPIPS}"
else
    echo "[!] Photometric metrics not found. Run Photo-SLAM-eval first."
fi
echo ""

# 2. Trajectory Accuracy (ATE)
echo "=== TRAJECTORY ACCURACY ==="
if [ -f "${RESULT_DIR}/ate_rmse.txt" ]; then
    ATE=$(cat "${RESULT_DIR}/ate_rmse.txt")
    echo "ATE RMSE: ${ATE} m"
elif [ -f "${RESULT_DIR}/CameraTrajectory_TUM.txt" ]; then
    echo "[i] ATE not computed yet. Run Photo-SLAM-eval/run.py first."
else
    echo "[!] Trajectory file not found."
fi
echo ""

# 3. Geometric Metrics (Mesh)
echo "=== GEOMETRIC METRICS ==="
# Find the depth directory (ends with _shutdown)
DEPTH_DIR=$(ls -d ${RESULT_DIR}/*_shutdown/depth 2>/dev/null | head -1)
MESH_FILE="${RESULT_DIR}/meshes/${SCENE_NAME}_rendered_full.ply"

if [ -f "${MESH_FILE}" ]; then
    # Run mesh evaluation
    cd "${BASE_DIR}/neural_slam_eval-main"
    source "${TSDF_ENV_PATH}/bin/activate"
    
    EVAL_OUTPUT=$(python eval_recon.py \
        --rec_mesh "../${MESH_FILE}" \
        --gt_mesh "${GT_MESH}" \
        -3d 2>&1)
    
    ACC=$(echo "$EVAL_OUTPUT" | grep "accuracy:" | awk '{printf "%.4f", $2}')
    COMP=$(echo "$EVAL_OUTPUT" | grep "completion:" | awk '{printf "%.4f", $2}')
    COMP_RATIO=$(echo "$EVAL_OUTPUT" | grep "completion ratio:" | awk '{printf "%.2f", $3}')
    
    echo "Accuracy:         ${ACC} cm"
    echo "Completion:       ${COMP} cm"
    echo "Completion Ratio: ${COMP_RATIO}%"
    
    cd "$BASE_DIR"
else
    echo "[!] Mesh not found at ${MESH_FILE}"
    echo "    Generate mesh first with generate_full_room_mesh.py"
fi
echo ""

# 4. Summary Table
echo "=============================================="
echo "  SUMMARY TABLE"
echo "=============================================="
printf "| %-12s | %-12s |\n" "Metric" "Value"
echo "|--------------|--------------|"
if [ ! -z "$PSNR" ]; then
    printf "| %-12s | %-12s |\n" "PSNR" "${PSNR} dB"
    printf "| %-12s | %-12s |\n" "SSIM" "${SSIM}"
    printf "| %-12s | %-12s |\n" "LPIPS" "${LPIPS}"
fi
if [ ! -z "$ACC" ]; then
    printf "| %-12s | %-12s |\n" "Accuracy" "${ACC} cm"
    printf "| %-12s | %-12s |\n" "Completion" "${COMP} cm"
    printf "| %-12s | %-12s |\n" "Comp. Ratio" "${COMP_RATIO}%"
fi
echo "=============================================="
