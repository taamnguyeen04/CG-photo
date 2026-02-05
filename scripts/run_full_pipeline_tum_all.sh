#!/bin/bash
# ============================================================================
# Full Pipeline: Train + Evaluate Photo-SLAM / UncertPhoto-SLAM on TUM (ALL 3 SCENES)
# ============================================================================
#
# Usage:
#   ./run_full_pipeline_tum_all.sh <pa_name> <num_runs> [--uncer]
#
# Examples:
#   ./run_full_pipeline_tum_all.sh pa9 3              # All 3 TUM scenes x 3 runs (no uncertainty)
#   ./run_full_pipeline_tum_all.sh pa10 2             # All 3 TUM scenes x 2 runs (no uncertainty)
#   ./run_full_pipeline_tum_all.sh pa11_uncer 1 --uncer   # with uncertainty tracking enabled
#
# Options:
#   --uncer : Enable UncertPhoto-SLAM uncertainty tracking and pruning
#
# Automatically runs: freiburg1_desk, freiburg2_xyz, freiburg3_long_office_household
# ============================================================================

# Don't use set -e, we handle errors manually

# Parse arguments
PA_NAME="${1:?Usage: ./run_full_pipeline_tum_all.sh <pa_name> <num_runs> [--uncer]}"
NUM_RUNS="${2:?Missing number of runs}"
shift 2

# Check for --uncer flag
ENABLE_UNCER=false
if [ "$1" == "--uncer" ]; then
    ENABLE_UNCER=true
    shift
fi

# Fixed TUM scenes (short names used in configs)
SCENES=("freiburg1_desk" "freiburg2_xyz" "freiburg3_long_office_household")

# TUM scene folder names (full folder names in dataset)
declare -A TUM_FOLDERS
TUM_FOLDERS["freiburg1_desk"]="rgbd_dataset_freiburg1_desk"
TUM_FOLDERS["freiburg2_xyz"]="rgbd_dataset_freiburg2_xyz"
TUM_FOLDERS["freiburg3_long_office_household"]="rgbd_dataset_freiburg3_long_office_household"

# TUM associations file names
declare -A TUM_ASSOCIATIONS
TUM_ASSOCIATIONS["freiburg1_desk"]="tum_freiburg1_desk.txt"
TUM_ASSOCIATIONS["freiburg2_xyz"]="tum_freiburg2_xyz.txt"
TUM_ASSOCIATIONS["freiburg3_long_office_household"]="tum_freiburg3_long_office_household.txt"

# Paths
BASE_DIR="/media/tam/DATA/3D/CG-photo"
GT_DATA_BASE="/media/tam/DATA/data/TUM"
GT_MESH_BASE="/media/tam/DATA/data/TUM/gt_meshes"
VENV_PATH="/media/tam/DATA/3D/Photo-SLAM/venv"
TSDF_ENV_PATH="${BASE_DIR}/scripts/tsdf_env"

# Config selection based on uncertainty flag
if [ "$ENABLE_UNCER" = true ]; then
    CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd_uncertainty.yaml"
    echo "[UncertPhoto-SLAM] Uncertainty tracking ENABLED"
else
    CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
fi

cd "$BASE_DIR"

# Extract loss weights from config
LAMBDA_GEO=$(grep "Optimization.lambda_geo:" "$CONFIG_FILE" | awk '{print $2}')
LAMBDA_SMOOTH=$(grep "Optimization.lambda_smooth:" "$CONFIG_FILE" | awk '{print $2}')
LAMBDA_VAR=$(grep "Optimization.lambda_var:" "$CONFIG_FILE" | awk '{print $2}')
LAMBDA_ISO=$(grep "Optimization.lambda_iso:" "$CONFIG_FILE" | awk '{print $2}')
LAMBDA_ALIGN=$(grep "Optimization.lambda_align:" "$CONFIG_FILE" | awk '{print $2}')

# Extract uncertainty config if enabled
if [ "$ENABLE_UNCER" = true ]; then
    UNCER_THRESHOLD=$(grep "Uncertainty.threshold:" "$CONFIG_FILE" | awk '{print $2}')
    UNCER_TAU_OBS=$(grep "Uncertainty.tau_obs:" "$CONFIG_FILE" | awk '{print $2}')
fi

echo "=============================================="
echo "  Full Pipeline: ${PA_NAME} - TUM ALL SCENES"
echo "  Scenes: ${SCENES[*]}"
echo "  Runs per scene: ${NUM_RUNS}"
echo "  Uncertainty: ${ENABLE_UNCER}"
echo "=============================================="
echo ""
echo "=== LOSS WEIGHTS (from config) ==="
echo "lambda_geo:    ${LAMBDA_GEO}"
echo "lambda_smooth: ${LAMBDA_SMOOTH}"
echo "lambda_var:    ${LAMBDA_VAR}"
echo "lambda_iso:    ${LAMBDA_ISO}"
echo "lambda_align:  ${LAMBDA_ALIGN}"
if [ "$ENABLE_UNCER" = true ]; then
    echo ""
    echo "=== UNCERTAINTY CONFIG ==="
    echo "threshold:     ${UNCER_THRESHOLD}"
    echo "tau_obs:       ${UNCER_TAU_OBS}"
fi
echo ""

# Create results summary file (append mode - only add header if file doesn't exist)
SUMMARY_ALL="${BASE_DIR}/results_${PA_NAME}/all_results.csv"
FAILED_LOG="${BASE_DIR}/results_${PA_NAME}/failed_runs.txt"
mkdir -p "${BASE_DIR}/results_${PA_NAME}"

# Only write header if CSV doesn't exist or is empty
if [ ! -f "$SUMMARY_ALL" ] || [ ! -s "$SUMMARY_ALL" ]; then
    if [ "$ENABLE_UNCER" = true ]; then
        echo "scene,run,psnr,ssim,lpips,ate_rmse,accuracy,completion,comp_ratio,chamfer,gauss_init,gauss_final,gauss_pruned" > "$SUMMARY_ALL"
    else
        echo "scene,run,psnr,ssim,lpips,ate_rmse,accuracy,completion,comp_ratio,chamfer" > "$SUMMARY_ALL"
    fi
fi
echo "# Failed runs log - $(date)" >> "$FAILED_LOG"

# Function to run single scene
run_single_scene() {
    local SCENE=$1
    local RUN=$2
    local SCENE_FOLDER="${TUM_FOLDERS[$SCENE]}"
    local SCENE_ASSOC="${TUM_ASSOCIATIONS[$SCENE]}"
    local RESULT_DIR="${BASE_DIR}/results_${PA_NAME}/${SCENE}_run${RUN}"
    local GT_DATA_DIR="${GT_DATA_BASE}/${SCENE_FOLDER}"
    local GT_MESH="${GT_MESH_BASE}/${SCENE}_gt.ply"
    local ORB_CONFIG="${BASE_DIR}/cfg/ORB_SLAM3/RGB-D/TUM/tum_${SCENE}.yaml"
    local SCENE_CONFIG="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_${SCENE}.yaml"
    
    # Use uncertainty config if enabled
    if [ "$ENABLE_UNCER" = true ]; then
        SCENE_CONFIG="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd_uncertainty.yaml"
    fi
    
    echo ""
    echo "======================================================"
    echo "  Running: ${SCENE} (Run ${RUN}/${NUM_RUNS})"
    if [ "$ENABLE_UNCER" = true ]; then
        echo "  [UncertPhoto-SLAM Mode]"
    fi
    echo "======================================================"
    
    # Delete existing results
    if [ -d "$RESULT_DIR" ]; then
        echo "[!] Deleting existing: $RESULT_DIR"
        rm -rf "$RESULT_DIR"
    fi
    
    # Step 1: Training
    echo "--- Training ---"
    cd "$BASE_DIR"
    if ! ./bin/tum_rgbd \
        ORB-SLAM3/Vocabulary/ORBvoc.txt \
        "$ORB_CONFIG" \
        "$SCENE_CONFIG" \
        "${GT_DATA_DIR}" \
        "cfg/ORB_SLAM3/RGB-D/TUM/associations/${SCENE_ASSOC}" \
        "results_${PA_NAME}/${SCENE}_run${RUN}" \
        no_viewer; then
        echo "[X] TRAINING FAILED: ${SCENE} run ${RUN}"
        echo "${SCENE},${RUN},FAILED,training" >> "$FAILED_LOG"
        echo "${SCENE},${RUN},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED" >> "$SUMMARY_ALL"
        return 1
    fi
    
    # Log uncertainty pruning stats if enabled
    INITIAL_GAUSS=""
    FINAL_GAUSS=""
    TOTAL_PRUNED=""
    if [ "$ENABLE_UNCER" = true ]; then
        PRUNE_LOG="${RESULT_DIR}/uncertainty_pruning.csv"
        if [ -f "$PRUNE_LOG" ]; then
            echo "--- Uncertainty Pruning Summary ---"
            INITIAL_GAUSS=$(head -2 "$PRUNE_LOG" | tail -1 | cut -d',' -f2)
            FINAL_GAUSS=$(tail -1 "$PRUNE_LOG" | cut -d',' -f3)
            TOTAL_PRUNED=$(awk -F',' 'NR>1 {sum+=$4} END {print sum}' "$PRUNE_LOG")
            echo "Initial Gaussians: ${INITIAL_GAUSS}"
            echo "Final Gaussians: ${FINAL_GAUSS}"
            echo "Total pruned: ${TOTAL_PRUNED}"
        fi
    fi
    
    # Step 2: Photometric Evaluation
    echo "--- Photometric Evaluation ---"
    cd "${BASE_DIR}/Photo-SLAM-eval"
    source "${VENV_PATH}/bin/activate"
    python run.py "../results_${PA_NAME}/${SCENE}_run${RUN}" "${GT_DATA_DIR}"
    
    # Calculate metrics
    PSNR=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/psnr.txt")
    SSIM=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/ssim.txt")
    LPIPS=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/lpips.txt")
    
    # Extract ATE RMSE from metrics_traj.txt (trajectory error in meters)
    ATE_RMSE="N/A"
    if [ -f "${RESULT_DIR}/metrics_traj.txt" ]; then
        ATE_RMSE=$(grep -m1 "rmse" "${RESULT_DIR}/metrics_traj.txt" | awk '{printf "%.6f", $2}')
    fi
    
    # Step 3: Generate Mesh (using cameras.json for correct coordinate alignment)
    echo "--- Generating Mesh ---"
    cd "$BASE_DIR"
    source "${TSDF_ENV_PATH}/bin/activate"
    
    # Find shutdown directory containing cameras.json
    SHUTDOWN_DIR=$(ls -d ${RESULT_DIR}/*_shutdown 2>/dev/null | head -1)
    JSON_PATH="${SHUTDOWN_DIR}/ply/cameras.json"
    DEPTH_DIR="${SHUTDOWN_DIR}/depth"
    
    # TUM groundtruth file for alignment
    GT_TRAJ_TUM="${GT_DATA_DIR}/groundtruth.txt"
    EST_TRAJ_TUM="${RESULT_DIR}/CameraTrajectory_TUM.txt"
    
    mkdir -p "${RESULT_DIR}/meshes"
    
    # Initialize geometry metrics
    ACC="N/A"
    COMP="N/A"
    COMP_RATIO="N/A"
    CHAMFER="N/A"
    
    if [ -f "$JSON_PATH" ]; then
        # Generate mesh using cameras.json with TUM groundtruth for alignment
        python scripts/generate_mesh_from_json.py \
            --json_path "$JSON_PATH" \
            --depth_dir "$DEPTH_DIR" \
            --output "${RESULT_DIR}/meshes/${SCENE}_json_aligned.ply" \
            --voxel_size 0.01 \
            --depth_scale 5000.0 \
            --max_depth 5.0 \
            --gt_traj_tum "$GT_TRAJ_TUM" \
            --est_traj_tum "$EST_TRAJ_TUM"
        MESH_FILE="${RESULT_DIR}/meshes/${SCENE}_json_aligned.ply"
        
        # Step 4: Geometric Evaluation (only if GT mesh exists)
        if [ -f "$GT_MESH" ]; then
            echo "--- Geometric Evaluation ---"
            cd "${BASE_DIR}/neural_slam_eval-main"
            EVAL_OUTPUT=$(python eval_recon.py \
                --rec_mesh "$MESH_FILE" \
                --gt_mesh "${GT_MESH}" \
                -3d 2>&1)
            
            ACC=$(echo "$EVAL_OUTPUT" | grep "accuracy:" | awk '{printf "%.4f", $2}')
            COMP=$(echo "$EVAL_OUTPUT" | grep "completion:" | awk '{printf "%.4f", $2}')
            COMP_RATIO=$(echo "$EVAL_OUTPUT" | grep "completion ratio:" | awk '{printf "%.2f", $3}')
            CHAMFER=$(echo "$EVAL_OUTPUT" | grep "chamfer:" | awk '{printf "%.4f", $2}')
        else
            echo "[!] GT mesh not found: $GT_MESH - Skipping geometric evaluation"
            echo "[!] Run scripts/generate_tum_gt_mesh.py --all to generate GT meshes"
        fi
    else
        echo "[!] cameras.json not found: $JSON_PATH - Skipping mesh generation"
    fi
    
    # Print summary for this run
    echo ""
    echo "--- ${SCENE} Run ${RUN} Results ---"
    echo "PSNR: ${PSNR} | SSIM: ${SSIM} | LPIPS: ${LPIPS} | ATE: ${ATE_RMSE}m"
    echo "Acc: ${ACC}cm | Comp: ${COMP}cm | Chamfer: ${CHAMFER}cm | Ratio: ${COMP_RATIO}%"
    if [ -n "$INITIAL_GAUSS" ] && [ -n "$FINAL_GAUSS" ]; then
        echo "Gaussians: ${INITIAL_GAUSS} -> ${FINAL_GAUSS} (pruned: ${TOTAL_PRUNED})"
    fi
    
    # Append to CSV (add Gaussian stats if uncertainty enabled)
    if [ "$ENABLE_UNCER" = true ] && [ -n "$INITIAL_GAUSS" ]; then
        echo "${SCENE},${RUN},${PSNR},${SSIM},${LPIPS},${ATE_RMSE},${ACC},${COMP},${COMP_RATIO},${CHAMFER},${INITIAL_GAUSS},${FINAL_GAUSS},${TOTAL_PRUNED}" >> "$SUMMARY_ALL"
    else
        echo "${SCENE},${RUN},${PSNR},${SSIM},${LPIPS},${ATE_RMSE},${ACC},${COMP},${COMP_RATIO},${CHAMFER}" >> "$SUMMARY_ALL"
    fi
    
    # Save individual summary
    cat > "${RESULT_DIR}/summary.txt" <<EOF
PA: ${PA_NAME}, Scene: ${SCENE}, Run: ${RUN}
Uncertainty Enabled: ${ENABLE_UNCER}

# Loss Weights
lambda_geo: ${LAMBDA_GEO}
lambda_smooth: ${LAMBDA_SMOOTH}
lambda_var: ${LAMBDA_VAR}
lambda_iso: ${LAMBDA_ISO}
lambda_align: ${LAMBDA_ALIGN}

# Photometric Metrics
PSNR: ${PSNR}
SSIM: ${SSIM}
LPIPS: ${LPIPS}

# Geometric Metrics
Accuracy: ${ACC}
Completion: ${COMP}
Completion_Ratio: ${COMP_RATIO}
Chamfer: ${CHAMFER}
EOF
}

# Main loop
TOTAL_RUNS=$((${#SCENES[@]} * NUM_RUNS))
CURRENT=0

for SCENE in "${SCENES[@]}"; do
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "######################################################"
        echo "  Progress: ${CURRENT}/${TOTAL_RUNS}"
        echo "######################################################"
        run_single_scene "$SCENE" "$RUN"
    done
done

# Final Summary
cd "$BASE_DIR"
echo ""
echo "=============================================="
echo "  ALL TUM RESULTS: ${PA_NAME}"
if [ "$ENABLE_UNCER" = true ]; then
    echo "  [UncertPhoto-SLAM Mode]"
fi
echo "=============================================="
echo ""
cat "$SUMMARY_ALL" | column -t -s','
echo ""
echo "[✓] All results saved to: ${SUMMARY_ALL}"
