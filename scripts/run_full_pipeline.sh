#!/bin/bash
# Full pipeline: Train + Evaluate Photo-SLAM on Replica (Multiple scenes, multiple runs)
# 
# Supports: RGB-D, Monocular, Stereo modes
#
# Usage:
#   ./run_full_pipeline.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ...
#
# Modes: rgbd, mono, stereo
#
# Examples:
#   ./run_full_pipeline.sh rgbd pa9 2 office0 room0       # RGB-D mode
#   ./run_full_pipeline.sh mono baseline 1 office0       # Monocular mode
#   ./run_full_pipeline.sh stereo pa1 1 office0          # Stereo mode


# Parse arguments
MODE="${1:?Usage: ./run_full_pipeline.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ...}"
PA_NAME="${2:?Missing PA name}"
NUM_RUNS="${3:?Missing number of runs}"
shift 3
SCENES=("$@")

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No scenes specified!"
    echo "Usage: ./run_full_pipeline.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ..."
    exit 1
fi

# Validate mode
MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
if [[ ! "$MODE" =~ ^(rgbd|mono|stereo)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be: rgbd, mono, stereo"
    exit 1
fi

# Paths - Replica dataset
BASE_DIR="/media/tam/DATA/3D/CG-photo"
GT_DATA_BASE="/media/tam/DATA/data/Replica"
GT_MESH_BASE="/media/tam/DATA/data/Replica/cull_replica_mesh"
VENV_PATH="/media/tam/DATA/3D/Photo-SLAM/venv"
TSDF_ENV_PATH="${BASE_DIR}/scripts/tsdf_env"
DATASET="replica"

cd "$BASE_DIR"

# Set config and binary based on mode
case "$MODE" in
    rgbd)
        MODE_UPPER="RGB-D"
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
        ORB_CONFIG_BASE="${BASE_DIR}/cfg/ORB_SLAM3/RGB-D/Replica"
        BINARY="./bin/replica_rgbd"
        DEPTH_SCALE=6553.5
        ;;
    mono)
        MODE_UPPER="Monocular"
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
        ORB_CONFIG_BASE="${BASE_DIR}/cfg/ORB_SLAM3/Monocular/Replica"
        BINARY="./bin/replica_mono"
        DEPTH_SCALE=6553.5
        ;;
    stereo)
        MODE_UPPER="Stereo"
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/Stereo/Replica/replica_stereo.yaml"
        ORB_CONFIG_BASE="${BASE_DIR}/cfg/ORB_SLAM3/Stereo/Replica"
        BINARY="./bin/replica_stereo"
        DEPTH_SCALE=6553.5
        ;;
esac

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    echo "Available binaries:"
    ls -1 ./bin/ 2>/dev/null
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract loss weights from config
LAMBDA_GEO=$(grep "Optimization.lambda_geo:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')
LAMBDA_SMOOTH=$(grep "Optimization.lambda_smooth:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')
LAMBDA_VAR=$(grep "Optimization.lambda_var:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')
LAMBDA_ISO=$(grep "Optimization.lambda_iso:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')
LAMBDA_ALIGN=$(grep "Optimization.lambda_align:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')

echo "=============================================="
echo "  Full Pipeline: ${PA_NAME}"
echo "  Mode: ${MODE_UPPER}"
echo "  Scenes: ${SCENES[*]}"
echo "  Runs per scene: ${NUM_RUNS}"
echo "=============================================="
echo ""
echo "=== LOSS WEIGHTS (from config) ==="
echo "lambda_geo:    ${LAMBDA_GEO:-0.0}"
echo "lambda_smooth: ${LAMBDA_SMOOTH:-0.0}"
echo "lambda_var:    ${LAMBDA_VAR:-0.0}"
echo "lambda_iso:    ${LAMBDA_ISO:-0.0}"
echo "lambda_align:  ${LAMBDA_ALIGN:-0.0}"
echo ""

# Create results directory with organized structure: results/{dataset}/{mode}/{pa_name}
RESULTS_BASE="${BASE_DIR}/results/${DATASET}/${MODE}"
RESULTS_DIR="${RESULTS_BASE}/${PA_NAME}"
SUMMARY_ALL="${RESULTS_DIR}/all_results.csv"
FAILED_LOG="${RESULTS_DIR}/failed_runs.txt"
mkdir -p "${RESULTS_DIR}"

# Only write header if CSV doesn't exist or is empty
if [ ! -f "$SUMMARY_ALL" ] || [ ! -s "$SUMMARY_ALL" ]; then
    echo "scene,run,psnr,ssim,lpips,ate_rmse,accuracy,completion,comp_ratio,chamfer" > "$SUMMARY_ALL"
fi
echo "# Failed runs log - $(date)" >> "$FAILED_LOG"

# Function to run single scene
run_single_scene() {
    local SCENE=$1
    local RUN=$2
    local RESULT_DIR="${RESULTS_DIR}/${SCENE}_run${RUN}"
    local GT_DATA_DIR="${GT_DATA_BASE}/${SCENE}"
    local GT_MESH="${GT_MESH_BASE}/${SCENE}.ply"
    local ORB_CONFIG="${ORB_CONFIG_BASE}/${SCENE}.yaml"
    
    echo ""
    echo "======================================================"
    echo "  Running: ${SCENE} (Run ${RUN}/${NUM_RUNS})"
    echo "  Mode: ${MODE_UPPER}"
    echo "======================================================"
    
    # Check ORB config
    if [ ! -f "$ORB_CONFIG" ]; then
        echo "[X] ORB config not found: $ORB_CONFIG"
        echo "${SCENE},${RUN},FAILED,orb_config_missing" >> "$FAILED_LOG"
        return 1
    fi
    
    # Delete existing results
    if [ -d "$RESULT_DIR" ]; then
        echo "[!] Deleting existing: $RESULT_DIR"
        rm -rf "$RESULT_DIR"
    fi
    
    # Step 1: Training
    echo "--- Training ---"
    cd "$BASE_DIR"
    if ! $BINARY \
        ORB-SLAM3/Vocabulary/ORBvoc.txt \
        "$ORB_CONFIG" \
        "$CONFIG_FILE" \
        "${GT_DATA_DIR}" \
        "results/${DATASET}/${MODE}/${PA_NAME}/${SCENE}_run${RUN}" \
        no_viewer; then
        echo "[X] TRAINING FAILED: ${SCENE} run ${RUN}"
        echo "${SCENE},${RUN},FAILED,training" >> "$FAILED_LOG"
        echo "${SCENE},${RUN},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED" >> "$SUMMARY_ALL"
        return 1
    fi
    
    # Step 2: Photometric Evaluation
    echo "--- Photometric Evaluation ---"
    cd "${BASE_DIR}/Photo-SLAM-eval"
    source "${VENV_PATH}/bin/activate"
    python run.py "${RESULT_DIR}" "${GT_DATA_DIR}"
    
    # Calculate photometric metrics
    PSNR=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/psnr.txt" 2>/dev/null || echo "N/A")
    SSIM=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/ssim.txt" 2>/dev/null || echo "N/A")
    LPIPS=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/lpips.txt" 2>/dev/null || echo "N/A")
    
    # Extract ATE RMSE from metrics_traj.txt (line 8 contains "rmse <value>")
    ATE_RMSE="N/A"
    if [ -f "${RESULT_DIR}/metrics_traj.txt" ]; then
        ATE_RMSE=$(sed -n '8p' "${RESULT_DIR}/metrics_traj.txt" | awk '{printf "%.6f", $2}')
    fi
    
    # Step 3: Generate Mesh (using cameras.json for correct coordinate alignment)
    echo "--- Generating Mesh ---"
    cd "$BASE_DIR"
    source "${TSDF_ENV_PATH}/bin/activate"
    
    # Find shutdown directory containing cameras.json
    SHUTDOWN_DIR=$(ls -d ${RESULT_DIR}/*_shutdown 2>/dev/null | head -1)
    JSON_PATH="${SHUTDOWN_DIR}/ply/cameras.json"
    DEPTH_DIR="${SHUTDOWN_DIR}/depth"
    GT_TRAJ="${GT_DATA_DIR}/traj.txt"
    
    mkdir -p "${RESULT_DIR}/meshes"
    
    if [ ! -f "$JSON_PATH" ]; then
        echo "[X] ERROR: cameras.json not found: $JSON_PATH"
        echo "${SCENE},${RUN},ERROR,cameras.json not found" >> "$FAILED_LOG"
        return 1
    fi
    
    # Generate mesh using cameras.json with GT trajectory for coordinate alignment
    python scripts/generate_mesh_from_json.py \
        --json_path "$JSON_PATH" \
        --depth_dir "$DEPTH_DIR" \
        --output "${RESULT_DIR}/meshes/${SCENE}_json_aligned.ply" \
        --voxel_size 0.01 \
        --depth_scale $DEPTH_SCALE \
        --max_depth 10.0 \
        --gt_traj "$GT_TRAJ"
    MESH_FILE="${RESULT_DIR}/meshes/${SCENE}_json_aligned.ply"
    
    # Step 4: Geometric Evaluation
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
    
    # Print summary for this run
    echo ""
    echo "--- ${SCENE} Run ${RUN} Results ---"
    echo "PSNR: ${PSNR} | SSIM: ${SSIM} | LPIPS: ${LPIPS} | ATE: ${ATE_RMSE}m"
    echo "Acc: ${ACC}cm | Comp: ${COMP}cm | Chamfer: ${CHAMFER}cm | Ratio: ${COMP_RATIO}%"
    
    # Append to CSV
    echo "${SCENE},${RUN},${PSNR},${SSIM},${LPIPS},${ATE_RMSE},${ACC},${COMP},${COMP_RATIO},${CHAMFER}" >> "$SUMMARY_ALL"
    
    # Save individual summary
    cat > "${RESULT_DIR}/summary.txt" << EOF
PA: ${PA_NAME}, Scene: ${SCENE}, Run: ${RUN}
Mode: ${MODE_UPPER}

# Loss Weights
lambda_geo: ${LAMBDA_GEO:-0.0}
lambda_smooth: ${LAMBDA_SMOOTH:-0.0}
lambda_var: ${LAMBDA_VAR:-0.0}
lambda_iso: ${LAMBDA_ISO:-0.0}
lambda_align: ${LAMBDA_ALIGN:-0.0}

# Metrics
PSNR: ${PSNR}
SSIM: ${SSIM}
LPIPS: ${LPIPS}
ATE_RMSE: ${ATE_RMSE}
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
echo "  ALL RESULTS: ${PA_NAME} (${MODE_UPPER})"
echo "=============================================="
echo ""
cat "$SUMMARY_ALL" | column -t -s','
echo ""
echo "[✓] All results saved to: ${SUMMARY_ALL}"
