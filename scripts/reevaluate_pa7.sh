#!/bin/bash
# Re-evaluate all PA7 results (Photometric + Geometric with Chamfer Distance)
# Usage: ./scripts/reevaluate_pa7.sh

set -e

BASE_DIR="/media/tam/DATA/3D/CG-photo"
GT_DATA_BASE="/media/tam/DATA/data/Replica"
GT_MESH_BASE="/media/tam/DATA/data/Replica/cull_replica_mesh"
VENV_PATH="/media/tam/DATA/3D/Photo-SLAM/venv"
TSDF_ENV_PATH="${BASE_DIR}/scripts/tsdf_env"
PA_NAME="pa7"

cd "$BASE_DIR"

# Output file
OUTPUT_CSV="${BASE_DIR}/results_${PA_NAME}/all_results_reevaluated.csv"
echo "scene,run,psnr,ssim,lpips,accuracy,completion,comp_ratio,chamfer" > "$OUTPUT_CSV"

# All scenes and runs
SCENES=(office0 office1 office2 office3 office4 room0 room1 room2)
RUNS=(1 2)

for SCENE in "${SCENES[@]}"; do
    for RUN in "${RUNS[@]}"; do
        RESULT_DIR="${BASE_DIR}/results_${PA_NAME}/${SCENE}_run${RUN}"
        GT_DATA_DIR="${GT_DATA_BASE}/${SCENE}"
        GT_MESH="${GT_MESH_BASE}/${SCENE}.ply"
        
        if [ ! -d "$RESULT_DIR" ]; then
            echo "[SKIP] $RESULT_DIR not found"
            continue
        fi
        
        echo ""
        echo "========================================"
        echo "  Evaluating: ${SCENE}_run${RUN}"
        echo "========================================"
        
        # Step 1: Photometric Metrics (from existing files)
        PSNR="N/A"
        SSIM="N/A"
        LPIPS="N/A"
        
        if [ -f "${RESULT_DIR}/psnr.txt" ]; then
            PSNR=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/psnr.txt")
            SSIM=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/ssim.txt")
            LPIPS=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/lpips.txt")
            echo "  Photometric: PSNR=${PSNR} SSIM=${SSIM} LPIPS=${LPIPS}"
        else
            echo "  [!] Photometric files not found, running evaluation..."
            cd "${BASE_DIR}/Photo-SLAM-eval"
            source "${VENV_PATH}/bin/activate"
            python run.py "../results_${PA_NAME}/${SCENE}_run${RUN}" "${GT_DATA_DIR}"
            cd "$BASE_DIR"
            PSNR=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/psnr.txt")
            SSIM=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/ssim.txt")
            LPIPS=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/lpips.txt")
        fi
        
        # Step 2: Check if mesh exists, generate if not
        MESH_FILE="${RESULT_DIR}/meshes/${SCENE}_json_aligned.ply"
        
        if [ ! -f "$MESH_FILE" ]; then
            echo "  [!] Mesh not found, generating..."
            source "${TSDF_ENV_PATH}/bin/activate"
            SHUTDOWN_DIR=$(ls -d ${RESULT_DIR}/*_shutdown 2>/dev/null | head -1)
            JSON_PATH="${SHUTDOWN_DIR}/ply/cameras.json"
            DEPTH_DIR="${SHUTDOWN_DIR}/depth"
            GT_TRAJ="${GT_DATA_DIR}/traj.txt"
            
            mkdir -p "${RESULT_DIR}/meshes"
            
            python scripts/generate_mesh_from_json.py \
                --json_path "$JSON_PATH" \
                --depth_dir "$DEPTH_DIR" \
                --output "$MESH_FILE" \
                --voxel_size 0.01 \
                --depth_scale 6553.5 \
                --max_depth 10.0 \
                --gt_traj "$GT_TRAJ"
        fi
        
        # Step 3: Geometric Evaluation with Chamfer
        echo "  Running geometric evaluation..."
        cd "${BASE_DIR}/neural_slam_eval-main"
        source "${TSDF_ENV_PATH}/bin/activate"
        
        EVAL_OUTPUT=$(python eval_recon.py \
            --rec_mesh "$MESH_FILE" \
            --gt_mesh "${GT_MESH}" \
            -3d 2>&1)
        
        ACC=$(echo "$EVAL_OUTPUT" | grep "accuracy:" | awk '{printf "%.4f", $2}')
        COMP=$(echo "$EVAL_OUTPUT" | grep "completion:" | awk '{printf "%.4f", $2}')
        COMP_RATIO=$(echo "$EVAL_OUTPUT" | grep "completion ratio:" | awk '{printf "%.2f", $3}')
        CHAMFER=$(echo "$EVAL_OUTPUT" | grep "chamfer:" | awk '{printf "%.4f", $2}')
        
        echo "  Geometric: Acc=${ACC}cm Comp=${COMP}cm Chamfer=${CHAMFER}cm Ratio=${COMP_RATIO}%"
        
        # Append to CSV
        echo "${SCENE},${RUN},${PSNR},${SSIM},${LPIPS},${ACC},${COMP},${COMP_RATIO},${CHAMFER}" >> "$OUTPUT_CSV"
        
        cd "$BASE_DIR"
    done
done

echo ""
echo "========================================"
echo "  ALL RESULTS"
echo "========================================"
cat "$OUTPUT_CSV" | column -t -s','
echo ""
echo "[✓] Results saved to: $OUTPUT_CSV"
