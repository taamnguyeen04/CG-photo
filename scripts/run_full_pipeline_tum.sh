#!/bin/bash
# Full pipeline: Train + Evaluate Photo-SLAM on TUM RGB-D Dataset (Photometric Only)
# 
# Supports both RGB-D and Monocular modes
#
# Usage:
#   ./scripts/run_full_pipeline_tum.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ...
#
# Modes: rgbd, mono
#
# Scenes (short aliases supported):
#   fr1_desk   → freiburg1_desk
#   fr1_room   → freiburg1_room
#   fr1_360    → freiburg1_360
#   fr1_rpy    → freiburg1_rpy
#   fr2_xyz    → freiburg2_xyz
#   fr2_360h   → freiburg2_360_hemisphere
#   fr3_office → freiburg3_long_office_household
#   fr3_str_tex_near → freiburg3_structure_texture_near
#   fr3_str_tex_far  → freiburg3_structure_texture_far
#   fr3_str_notex_near → freiburg3_structure_notexture_near
#   fr3_str_notex_far  → freiburg3_structure_notexture_far
#   fr3_nostr_tex_near → freiburg3_nostructure_texture_near_withloop
#   fr3_nostr_tex_far  → freiburg3_nostructure_texture_far
#   fr3_nostr_notex_near → freiburg3_nostructure_notexture_near_withloop
#   fr3_nostr_notex_far  → freiburg3_nostructure_notexture_far
#   (or use full names like freiburg1_desk)
#
# Examples:
#   ./scripts/run_full_pipeline_tum.sh rgbd test1 1 fr1_desk fr3_office
#   ./scripts/run_full_pipeline_tum.sh mono baseline 2 fr2_xyz
#   ./scripts/run_full_pipeline_tum.sh rgbd test_full 1 full


# ============================================================
# Alias expansion: short name → full scene name
# ============================================================
expand_alias() {
    local ALIAS=$1
    case "$ALIAS" in
        # Freiburg 1
        fr1_desk)   echo "freiburg1_desk" ;;
        fr1_room)   echo "freiburg1_room" ;;
        fr1_360)    echo "freiburg1_360" ;;
        fr1_rpy)    echo "freiburg1_rpy" ;;
        # Freiburg 2
        fr2_xyz)    echo "freiburg2_xyz" ;;
        fr2_360h)   echo "freiburg2_360_hemisphere" ;;
        fr2_360k)   echo "freiburg2_360_kidnap" ;;
        fr2_large)  echo "freiburg2_large_no_loop" ;;
        fr2_pioneer) echo "freiburg2_pioneer_slam" ;;
        # Freiburg 3
        fr3_office) echo "freiburg3_long_office_household" ;;
        fr3_str_tex_near)    echo "freiburg3_structure_texture_near" ;;
        fr3_str_tex_far)     echo "freiburg3_structure_texture_far" ;;
        fr3_str_notex_near)  echo "freiburg3_structure_notexture_near" ;;
        fr3_str_notex_far)   echo "freiburg3_structure_notexture_far" ;;
        fr3_nostr_tex_near)  echo "freiburg3_nostructure_texture_near_withloop" ;;
        fr3_nostr_tex_far)   echo "freiburg3_nostructure_texture_far" ;;
        fr3_nostr_notex_near) echo "freiburg3_nostructure_notexture_near_withloop" ;;
        fr3_nostr_notex_far)  echo "freiburg3_nostructure_notexture_far" ;;
        *)          echo "$ALIAS" ;;  # Pass through if already full name
    esac
}

# ============================================================
# Map scene name to TUM folder name
# ============================================================
get_tum_folder() {
    local SCENE=$1
    echo "rgbd_dataset_${SCENE}"
}

# ============================================================
# Auto-detect camera YAML from freiburg number
# ============================================================
get_camera_yaml() {
    local SCENE=$1
    local FR_NUM=$(echo "$SCENE" | grep -oP 'freiburg\K[0-9]')
    if [ -z "$FR_NUM" ]; then
        echo ""
        return 1
    fi
    echo "${GT_DATA_BASE}/camera_freiburg${FR_NUM}.yaml"
}

# ============================================================
# Get ORB-SLAM3 config
# ============================================================
get_orb_config() {
    local SCENE=$1
    local CONFIG="${ORB_CONFIG_BASE}/tum_${SCENE}.yaml"
    if [ -f "$CONFIG" ]; then
        echo "$CONFIG"
    else
        CONFIG="${ORB_CONFIG_BASE}/${SCENE}.yaml"
        if [ -f "$CONFIG" ]; then
            echo "$CONFIG"
        else
            # Fallback: detect freiburg number and use TUM{N}.yaml
            local FR_NUM=$(echo "$SCENE" | grep -oP 'freiburg\K[0-9]')
            echo "${ORB_CONFIG_BASE}/TUM${FR_NUM}.yaml"
        fi
    fi
}

# ============================================================
# Paths
# ============================================================
BASE_DIR="/media/tam/DATA/3D/CG-photo"
GT_DATA_BASE="/media/tam/DATA/data/TUM"
VENV_PATH="/media/tam/DATA/3D/Photo-SLAM/venv"

# ============================================================
# Parse arguments
# ============================================================
MODE="${1:?Usage: ./scripts/run_full_pipeline_tum.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ...}"
PA_NAME="${2:?Missing PA name}"
NUM_RUNS="${3:?Missing number of runs}"
shift 3

# Expand aliases
for ARG in "$@"; do
    if [ "$ARG" == "full" ]; then
        echo "Auto-detecting all scenes in ${GT_DATA_BASE}..."
        for DIR in "${GT_DATA_BASE}/rgbd_dataset_"*; do
             if [ -d "$DIR" ]; then
                 BASENAME=$(basename "$DIR")
                 # Remove prefix 'rgbd_dataset_' to get the scene name
                 SCENE_NAME=${BASENAME#rgbd_dataset_}
                 SCENES+=("$SCENE_NAME")
             fi
        done
        # Sort scenes for consistent order (optional but good)
        IFS=$'\n' SCENES=($(sort <<<"${SCENES[*]}"))
        unset IFS
    else
        SCENES+=("$(expand_alias "$ARG")")
    fi
done

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No scenes specified!"
    echo "Usage: ./scripts/run_full_pipeline_tum.sh <mode> <pa_name> <num_runs> <scene1> [scene2] ..."
    echo ""
    echo "Short aliases (examples):"
    echo "  fr1_desk   → freiburg1_desk"
    echo "  fr2_xyz    → freiburg2_xyz"
    echo "  fr3_office → freiburg3_long_office_household"
    echo ""
    echo "  Or use full scene names: freiburg1_desk, freiburg2_xyz, etc."
    exit 1
fi

# Validate mode
MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
if [[ ! "$MODE" =~ ^(rgbd|mono)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be: rgbd, mono"
    exit 1
fi

# ============================================================
# Paths (Defined at top)
# ============================================================
# BASE_DIR, GT_DATA_BASE, VENV_PATH defined above

cd "$BASE_DIR"

# Set paths based on mode
case "$MODE" in
    rgbd)
        MODE_UPPER="RGB-D"
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
        ORB_CONFIG_BASE="${BASE_DIR}/cfg/ORB_SLAM3/RGB-D/TUM"
        BINARY="./bin/tum_rgbd"
        ;;
    mono)
        MODE_UPPER="Monocular"
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml"
        ORB_CONFIG_BASE="${BASE_DIR}/cfg/ORB_SLAM3/Monocular/TUM"
        BINARY="./bin/tum_mono"
        ;;
esac

# Check binary
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found: $BINARY"
    exit 1
fi

# Check config
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
echo "=============================================="
echo ""
echo "=== LOSS WEIGHTS (from config) ==="
echo "lambda_geo:    ${LAMBDA_GEO:-0.0}"
echo "lambda_smooth: ${LAMBDA_SMOOTH:-0.0}"
echo "lambda_var:    ${LAMBDA_VAR:-0.0}"
echo "lambda_iso:    ${LAMBDA_ISO:-0.0}"
echo "lambda_align:  ${LAMBDA_ALIGN:-0.0}"
echo ""

# ============================================================
# Results directory
# ============================================================
DATASET="tum"
RESULTS_BASE="${BASE_DIR}/results/${DATASET}/${MODE}"
RESULTS_DIR="${RESULTS_BASE}/${PA_NAME}"
SUMMARY_ALL="${RESULTS_DIR}/all_results.csv"
FAILED_LOG="${RESULTS_DIR}/failed_runs.txt"
mkdir -p "$RESULTS_DIR"

# CSV header — photometric only + gaussians + fps
if [ ! -f "$SUMMARY_ALL" ] || [ ! -s "$SUMMARY_ALL" ]; then
    echo "scene,run,psnr,ssim,lpips,ate_rmse,gaussians,fps" > "$SUMMARY_ALL"
fi
echo "# Failed runs log - $(date)" >> "$FAILED_LOG"

# ============================================================
# Get association file for a scene
# ============================================================
get_association() {
    local SCENE=$1
    local GT_DATA_DIR=$2
    local ASSOC_DIR="${ORB_CONFIG_BASE}/associations"
    
    # Map full scene name → short association file name
    local SHORT_NAME=""
    case "$SCENE" in
        freiburg1_desk)   SHORT_NAME="fr1_desk" ;;
        freiburg1_room)   SHORT_NAME="fr1_room" ;;
        freiburg1_360)    SHORT_NAME="fr1_360" ;;
        freiburg1_rpy)    SHORT_NAME="fr1_rpy" ;;
        freiburg1_xyz)    SHORT_NAME="fr1_xyz" ;;
        freiburg2_xyz)    SHORT_NAME="fr2_xyz" ;;
        freiburg2_desk)   SHORT_NAME="fr2_desk" ;;
        freiburg2_360_hemisphere)   SHORT_NAME="fr2_360h" ;;
        freiburg2_360_kidnap)       SHORT_NAME="fr2_360k" ;;
        freiburg2_large_no_loop)    SHORT_NAME="fr2_large" ;;
        freiburg2_pioneer_slam)     SHORT_NAME="fr2_pioneer" ;;
        freiburg3_long_office_household) SHORT_NAME="fr3_office" ;;
        freiburg3_structure_texture_near)    SHORT_NAME="fr3_str_tex_near" ;;
        freiburg3_structure_texture_far)     SHORT_NAME="fr3_str_tex_far" ;;
        freiburg3_structure_notexture_near)  SHORT_NAME="fr3_str_notex_near" ;;
        freiburg3_structure_notexture_far)   SHORT_NAME="fr3_str_notex_far" ;;
        freiburg3_nostructure_texture_near_withloop)  SHORT_NAME="fr3_nstr_tex_near" ;;
        freiburg3_nostructure_texture_far)   SHORT_NAME="fr3_nstr_tex_far" ;;
        freiburg3_nostructure_notexture_near_withloop) SHORT_NAME="fr3_nstr_notex_near" ;;
        freiburg3_nostructure_notexture_far) SHORT_NAME="fr3_nstr_notex_far" ;;
    esac
    
    # Try: tum_{full_name}.txt, {short_name}.txt, {full_name}.txt in ORB associations dir
    for candidate in "tum_${SCENE}.txt" "${SHORT_NAME}.txt" "${SCENE}.txt"; do
        if [ -n "$candidate" ] && [ -f "${ASSOC_DIR}/${candidate}" ]; then
            echo "${ASSOC_DIR}/${candidate}"
            return 0
        fi
    done
    
    # Fallback: dataset dir
    if [ -f "${GT_DATA_DIR}/associations.txt" ]; then
        echo "${GT_DATA_DIR}/associations.txt"
        return 0
    fi
    
    # Not found
    echo ""
    return 1
}

# ============================================================
# Run single scene
# ============================================================
run_single_scene() {
    local SCENE=$1
    local RUN=$2
    local TUM_FOLDER=$(get_tum_folder "$SCENE")
    local GT_DATA_DIR="${GT_DATA_BASE}/${TUM_FOLDER}"
    local GT_TRAJ="${GT_DATA_DIR}/groundtruth.txt"
    local RESULT_DIR="${RESULTS_DIR}/${SCENE}_run${RUN}"
    local ORB_CONFIG=$(get_orb_config "$SCENE")
    local CAMERA_YAML=$(get_camera_yaml "$SCENE")
    
    # Find association file
    local ASSOC_FILE=$(get_association "$SCENE" "$GT_DATA_DIR")
    
    echo ""
    echo "======================================================"
    echo "  Running: ${SCENE} (Run ${RUN}/${NUM_RUNS})"
    echo "  Mode: ${MODE_UPPER}"
    echo "  TUM Folder: ${TUM_FOLDER}"
    echo "  Camera YAML: ${CAMERA_YAML}"
    echo "  Association: ${ASSOC_FILE}"
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
    
    # Check camera YAML
    if [ -z "$CAMERA_YAML" ] || [ ! -f "$CAMERA_YAML" ]; then
        echo "[X] Camera YAML not found: $CAMERA_YAML"
        echo "${SCENE},${RUN},FAILED,camera_yaml_missing" >> "$FAILED_LOG"
        return 1
    fi
    
    # Check ORB config
    if [ ! -f "$ORB_CONFIG" ]; then
        echo "[X] ORB config not found: $ORB_CONFIG"
        echo "    Available configs in ${ORB_CONFIG_BASE}:"
        ls -1 "$ORB_CONFIG_BASE" 2>/dev/null | head -10
        echo "${SCENE},${RUN},FAILED,orb_config_missing" >> "$FAILED_LOG"
        return 1
    fi
    
    # Check association file
    if [ -z "$ASSOC_FILE" ] || [ ! -f "$ASSOC_FILE" ]; then
        echo "[X] Association file not found for scene: $SCENE"
        echo "    Searched: ${ASSOC_DIR}/ and ${GT_DATA_DIR}/associations.txt"
        echo "    Available associations:"
        ls -1 "$ASSOC_DIR" 2>/dev/null | head -10
        echo "${SCENE},${RUN},FAILED,association_missing" >> "$FAILED_LOG"
        return 1
    fi
    
    # Delete existing results
    if [ -d "$RESULT_DIR" ]; then
        echo "[!] Deleting existing: $RESULT_DIR"
        rm -rf "$RESULT_DIR"
    fi
    
    # --- Step 1: Training ---
    echo "--- Training ---"
    cd "$BASE_DIR"
    if ! $BINARY \
        ORB-SLAM3/Vocabulary/ORBvoc.txt \
        "$ORB_CONFIG" \
        "$CONFIG_FILE" \
        "${GT_DATA_DIR}" \
        "$ASSOC_FILE" \
        "results/${DATASET}/${MODE}/${PA_NAME}/${SCENE}_run${RUN}" \
        no_viewer; then
        echo "[X] TRAINING FAILED: ${SCENE} run ${RUN}"
        echo "${SCENE},${RUN},FAILED,training" >> "$FAILED_LOG"
        echo "${SCENE},${RUN},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED" >> "$SUMMARY_ALL"
        return 1
    fi
    
    # --- Step 2: Photometric Evaluation ---
    echo "--- Photometric Evaluation ---"
    cd "${BASE_DIR}/Photo-SLAM-eval"
    source "${VENV_PATH}/bin/activate"
    python run.py "${RESULT_DIR}" "${GT_DATA_DIR}"
    
    # Calculate photometric metrics
    PSNR=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/psnr.txt" 2>/dev/null || echo "N/A")
    SSIM=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/ssim.txt" 2>/dev/null || echo "N/A")
    LPIPS=$(awk '{sum+=$1; count++} END {printf "%.4f", sum/count}' "${RESULT_DIR}/lpips.txt" 2>/dev/null || echo "N/A")
    
    # --- Step 3: Trajectory Evaluation (ATE RMSE) ---
    echo "--- Trajectory Evaluation ---"
    ATE_RMSE="N/A"
    
    # Check if run.py already generated metrics_traj.txt
    METRICS_FILE="${RESULT_DIR}/metrics_traj.txt"
    if [ -f "$METRICS_FILE" ]; then
        # Extract RMSE from the file (look for "rmse" and take 2nd column)
        # Use head -1 to get translation part (first block in file)
        ATE_RMSE=$(grep "rmse" "$METRICS_FILE" | head -1 | awk '{print $2}')
        if [ -z "$ATE_RMSE" ]; then ATE_RMSE="N/A"; fi
        echo "ATE RMSE (from eval): $ATE_RMSE"
    else
        echo "[!] metrics_traj.txt not found (trajectory eval skipped)"
    fi
    
    # --- Extract Gaussian count ---
    GAUSSIANS="N/A"
    if [ -f "${RESULT_DIR}/gaussian_count.txt" ]; then
        GAUSSIANS=$(cat "${RESULT_DIR}/gaussian_count.txt" | tr -d '[:space:]')
    fi
    
    # --- Calculate FPS from render_time.txt ---
    FPS="N/A"
    SHUTDOWN_DIR=$(ls -d ${RESULT_DIR}/*_shutdown 2>/dev/null | head -1)
    if [ -n "$SHUTDOWN_DIR" ]; then
        RENDER_TIME_FILE="${SHUTDOWN_DIR}/render_time.txt"
        if [ -f "$RENDER_TIME_FILE" ]; then
            # render_time.txt: each line has "kf_id time_ms", skip header (lines starting with ##)
            AVG_MS=$(grep -v '^##' "$RENDER_TIME_FILE" | awk '{sum+=$2; count++} END {if(count>0) printf "%.4f", sum/count; else print "0"}')
            if [ "$AVG_MS" != "0" ] && [ -n "$AVG_MS" ]; then
                FPS=$(echo "$AVG_MS" | awk '{printf "%.2f", 1000.0/$1}')
            fi
        fi
    fi
    
    # --- Print summary ---
    echo ""
    echo "--- ${SCENE} Run ${RUN} Results ---"
    echo "PSNR: ${PSNR} | SSIM: ${SSIM} | LPIPS: ${LPIPS}"
    echo "ATE RMSE: ${ATE_RMSE}m"
    echo "Gaussians: ${GAUSSIANS} | FPS: ${FPS}"
    
    # Append to CSV
    echo "${SCENE},${RUN},${PSNR},${SSIM},${LPIPS},${ATE_RMSE},${GAUSSIANS},${FPS}" >> "$SUMMARY_ALL"
    
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

# Model
Gaussians: ${GAUSSIANS}
FPS: ${FPS}
EOF
}

# ============================================================
# Main loop
# ============================================================
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

# ============================================================
# Final Summary
# ============================================================
cd "$BASE_DIR"
echo ""
echo "=============================================="
echo "  ALL RESULTS: ${PA_NAME} (TUM ${MODE_UPPER})"
echo "=============================================="
echo ""
cat "$SUMMARY_ALL" | column -t -s','
echo ""
echo "[✓] All results saved to: ${SUMMARY_ALL}"
