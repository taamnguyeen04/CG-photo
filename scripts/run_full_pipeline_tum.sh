#!/bin/bash
# Full pipeline: Train + Evaluate Photo-SLAM on TUM RGB-D Dataset
# 
# Supports both RGB-D and Monocular modes
#
# Usage:
#   ./run_full_pipeline_tum.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ...
#
# Modes: rgbd, mono
# Scenes: freiburg1_desk, freiburg2_xyz, freiburg3_long_office_household
#
# Examples:
#   ./run_full_pipeline_tum.sh rgbd geo07 2 freiburg1_desk
#   ./run_full_pipeline_tum.sh mono baseline 1 freiburg1_desk freiburg2_xyz


# Parse arguments
MODE="${1:?Usage: ./run_full_pipeline_tum.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ...}"
PA_NAME="${2:?Missing PA name}"
NUM_RUNS="${3:?Missing number of runs}"
shift 3
SCENES=("$@")

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No scenes specified!"
    echo "Usage: ./run_full_pipeline_tum.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ..."
    echo ""
    echo "Available scenes:"
    echo "  freiburg1_desk"
    echo "  freiburg2_xyz"
    echo "  freiburg3_long_office_household"
    exit 1
fi

# Validate mode
MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
if [[ ! "$MODE" =~ ^(rgbd|mono)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be: rgbd, mono"
    exit 1
fi

# Paths
BASE_DIR="/media/tam/DATA/3D/CG-photo"
GT_DATA_BASE="/media/tam/DATA/data/TUM"
GT_MESH_BASE="/media/tam/DATA/data/TUM/gt_meshes"
VENV_PATH="/media/tam/DATA/3D/Photo-SLAM/venv"
TSDF_ENV_PATH="${BASE_DIR}/scripts/tsdf_env"

cd "$BASE_DIR"

# Set paths based on mode
case "$MODE" in
    rgbd)
        MODE_UPPER="RGB-D"
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
        ORB_CONFIG_BASE="${BASE_DIR}/cfg/ORB_SLAM3/RGB-D/TUM"
        BINARY="./bin/tum_rgbd"
        HAS_GT_DEPTH=true
        DEPTH_SCALE=5000.0
        ;;
    mono)
        MODE_UPPER="Monocular"
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml"
        ORB_CONFIG_BASE="${BASE_DIR}/cfg/ORB_SLAM3/Monocular/TUM"
        BINARY="./bin/tum_mono"
        HAS_GT_DEPTH=false
        ;;
esac

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
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
echo "  Full Pipeline: ${PA_NAME} (TUM Dataset)"
echo "  Mode: ${MODE_UPPER}"
echo "  Scenes: ${SCENES[*]}"
echo "  Runs per scene: ${NUM_RUNS}"
echo "  Has GT Depth: ${HAS_GT_DEPTH}"
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
DATASET="tum"
RESULTS_BASE="${BASE_DIR}/results/${DATASET}/${MODE}"
RESULTS_DIR="${RESULTS_BASE}/${PA_NAME}"
SUMMARY_ALL="${RESULTS_DIR}/all_results.csv"
FAILED_LOG="${RESULTS_DIR}/failed_runs.txt"
mkdir -p "$RESULTS_DIR"

# CSV header - TUM has GT mesh so include geometric metrics
if [ ! -f "$SUMMARY_ALL" ] || [ ! -s "$SUMMARY_ALL" ]; then
    echo "scene,run,psnr,ssim,lpips,ate_rmse,accuracy,completion,comp_ratio,chamfer" > "$SUMMARY_ALL"
fi
echo "# Failed runs log - $(date)" >> "$FAILED_LOG"

# Map scene names to TUM folder names
get_tum_folder() {
    local SCENE=$1
    case "$SCENE" in
        freiburg1_desk)
            echo "rgbd_dataset_freiburg1_desk"
            ;;
        freiburg2_xyz)
            echo "rgbd_dataset_freiburg2_xyz"
            ;;
        freiburg3_long_office_household)
            echo "rgbd_dataset_freiburg3_long_office_household"
            ;;
        *)
            # Assume full folder name already given
            echo "$SCENE"
            ;;
    esac
}

# Map scene names to GT mesh names
get_gt_mesh() {
    local SCENE=$1
    case "$SCENE" in
        freiburg1_desk)
            echo "${GT_MESH_BASE}/freiburg1_desk_gt.ply"
            ;;
        freiburg2_xyz)
            echo "${GT_MESH_BASE}/freiburg2_xyz_gt.ply"
            ;;
        freiburg3_long_office_household)
            echo "${GT_MESH_BASE}/freiburg3_long_office_household_gt.ply"
            ;;
        *)
            echo "${GT_MESH_BASE}/${SCENE}_gt.ply"
            ;;
    esac
}

# Get ORB-SLAM3 config
get_orb_config() {
    local SCENE=$1
    # Use the specific config if exists, otherwise use generic
    local CONFIG="${ORB_CONFIG_BASE}/tum_${SCENE}.yaml"
    if [ -f "$CONFIG" ]; then
        echo "$CONFIG"
    else
        # Try alternative naming
        CONFIG="${ORB_CONFIG_BASE}/${SCENE}.yaml"
        if [ -f "$CONFIG" ]; then
            echo "$CONFIG"
        else
            # Use default TUM config
            echo "${ORB_CONFIG_BASE}/TUM1.yaml"
        fi
    fi
}

# Function to run single scene
run_single_scene() {
    local SCENE=$1
    local RUN=$2
    local TUM_FOLDER=$(get_tum_folder "$SCENE")
    local GT_DATA_DIR="${GT_DATA_BASE}/${TUM_FOLDER}"
    local GT_TRAJ="${GT_DATA_DIR}/groundtruth.txt"
    local GT_MESH=$(get_gt_mesh "$SCENE")
    local RESULT_DIR="${RESULTS_DIR}/${SCENE}_run${RUN}"
    local ORB_CONFIG=$(get_orb_config "$SCENE")
    
    echo ""
    echo "======================================================"
    echo "  Running: ${SCENE} (Run ${RUN}/${NUM_RUNS})"
    echo "  Mode: ${MODE_UPPER}"
    echo "  TUM Folder: ${TUM_FOLDER}"
    echo "======================================================"
    
    # Check GT data
    if [ ! -d "$GT_DATA_DIR" ]; then
        echo "[X] GT data not found: $GT_DATA_DIR"
        echo "${SCENE},${RUN},FAILED,gt_data_missing" >> "$FAILED_LOG"
        return 1
    fi
    
    # Check groundtruth
    if [ ! -f "$GT_TRAJ" ]; then
        echo "[!] Warning: groundtruth.txt not found: $GT_TRAJ"
    fi
    
    # Check ORB config
    if [ ! -f "$ORB_CONFIG" ]; then
        echo "[X] ORB config not found: $ORB_CONFIG"
        echo "    Available configs in ${ORB_CONFIG_BASE}:"
        ls -1 "$ORB_CONFIG_BASE" 2>/dev/null | head -10
        echo "${SCENE},${RUN},FAILED,orb_config_missing" >> "$FAILED_LOG"
        return 1
    fi
    
    # Check GT mesh
    HAS_GT_MESH=false
    if [ -f "$GT_MESH" ]; then
        HAS_GT_MESH=true
        echo "[✓] GT mesh found: $GT_MESH"
    else
        echo "[!] Warning: GT mesh not found: $GT_MESH"
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
    
    # Step 3: Trajectory Evaluation (ATE RMSE using groundtruth.txt)
    echo "--- Trajectory Evaluation ---"
    ATE_RMSE="N/A"
    if [ -f "$GT_TRAJ" ]; then
        # Find the CameraTrajectory file
        TRAJ_FILE="${RESULT_DIR}/CameraTrajectory_TUM.txt"
        if [ ! -f "$TRAJ_FILE" ]; then
            TRAJ_FILE=$(find "${RESULT_DIR}" -name "CameraTrajectory*.txt" -type f | head -1)
        fi
        
        if [ -f "$TRAJ_FILE" ]; then
            # Use evo_ate for TUM format trajectory evaluation
            if command -v evo_ate &> /dev/null; then
                ATE_OUTPUT=$(evo_ate tum "$GT_TRAJ" "$TRAJ_FILE" --align --correct_scale 2>&1)
                ATE_RMSE=$(echo "$ATE_OUTPUT" | grep "rmse" | awk '{printf "%.6f", $2}')
                if [ -z "$ATE_RMSE" ]; then
                    ATE_RMSE="N/A"
                fi
            else
                echo "[!] evo_ate not found, skipping trajectory evaluation"
            fi
        else
            echo "[!] Camera trajectory file not found"
        fi
    fi
    
    # Step 4 & 5: Mesh generation and evaluation (if GT mesh exists)
    ACC="N/A"
    COMP="N/A"
    COMP_RATIO="N/A"
    CHAMFER="N/A"
    
    if [ "$HAS_GT_MESH" = true ]; then
        echo "--- Generating Mesh ---"
        cd "$BASE_DIR"
        source "${TSDF_ENV_PATH}/bin/activate"
        
        SHUTDOWN_DIR=$(ls -d ${RESULT_DIR}/*_shutdown 2>/dev/null | head -1)
        JSON_PATH="${SHUTDOWN_DIR}/ply/cameras.json"
        DEPTH_DIR="${SHUTDOWN_DIR}/depth"
        
        mkdir -p "${RESULT_DIR}/meshes"
        
        if [ -f "$JSON_PATH" ] && [ -d "$DEPTH_DIR" ]; then
            python scripts/generate_mesh_from_json.py \
                --json_path "$JSON_PATH" \
                --depth_dir "$DEPTH_DIR" \
                --output "${RESULT_DIR}/meshes/${SCENE}_mesh.ply" \
                --voxel_size 0.01 \
                --depth_scale ${DEPTH_SCALE:-5000.0} \
                --max_depth 10.0
            MESH_FILE="${RESULT_DIR}/meshes/${SCENE}_mesh.ply"
            
            if [ -f "$MESH_FILE" ]; then
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
            fi
        else
            echo "[!] cameras.json or depth dir not found, skipping mesh generation"
        fi
    fi
    
    # Print summary for this run
    echo ""
    echo "--- ${SCENE} Run ${RUN} Results ---"
    echo "PSNR: ${PSNR} | SSIM: ${SSIM} | LPIPS: ${LPIPS}"
    echo "ATE RMSE: ${ATE_RMSE}m"
    if [ "$HAS_GT_MESH" = true ]; then
        echo "Acc: ${ACC}cm | Comp: ${COMP}cm | Chamfer: ${CHAMFER}cm | Ratio: ${COMP_RATIO}%"
    fi
    
    # Append to CSV
    echo "${SCENE},${RUN},${PSNR},${SSIM},${LPIPS},${ATE_RMSE},${ACC},${COMP},${COMP_RATIO},${CHAMFER}" >> "$SUMMARY_ALL"
    
    # Save individual summary
    cat > "${RESULT_DIR}/summary.txt" << EOF
PA: ${PA_NAME}, Scene: ${SCENE}, Run: ${RUN}
Mode: ${MODE_UPPER}
Dataset: TUM

# Loss Weights
lambda_geo: ${LAMBDA_GEO:-0.0}
lambda_smooth: ${LAMBDA_SMOOTH:-0.0}
lambda_var: ${LAMBDA_VAR:-0.0}
lambda_iso: ${LAMBDA_ISO:-0.0}
lambda_align: ${LAMBDA_ALIGN:-0.0}

# Photometric Metrics
PSNR: ${PSNR}
SSIM: ${SSIM}
LPIPS: ${LPIPS}

# Trajectory
ATE_RMSE: ${ATE_RMSE}

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
echo "  ALL RESULTS: ${PA_NAME} (TUM ${MODE_UPPER})"
echo "=============================================="
echo ""
cat "$SUMMARY_ALL" | column -t -s','
echo ""
echo "[✓] All results saved to: ${SUMMARY_ALL}"
