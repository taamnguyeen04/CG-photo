#!/bin/bash
# ============================================================================
# Multi-Experiment Pipeline for Photo-SLAM on TUM Dataset
# ============================================================================
# Run multiple experiments with different loss configurations on TUM.
#
# Usage:
#   ./run_multi_experiments_tum.sh <mode> <experiments_file> <num_runs> <scene1> [scene2] ...
#
# Modes: rgbd, mono
#
# Examples:
#   ./run_multi_experiments_tum.sh rgbd experiments.txt 2 freiburg1_desk
#   ./run_multi_experiments_tum.sh mono experiments.txt 1 freiburg1_desk freiburg2_xyz
# ============================================================================

set -e

# Parse arguments
MODE="${1:?Usage: ./run_multi_experiments_tum.sh <mode> <experiments_file> <num_runs> <scene1> [scene2] ...}"
EXPERIMENTS_FILE="${2:?Missing experiments file}"
NUM_RUNS="${3:?Missing number of runs}"
shift 3
SCENES=("$@")

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No scenes specified!"
    echo "Usage: ./run_multi_experiments_tum.sh <mode> <experiments_file> <num_runs> <scene1> [scene2] ..."
    echo ""
    echo "Available scenes (short names):"
    echo "  freiburg1_desk"
    echo "  freiburg2_xyz"
    echo "  freiburg3_long_office_household"
    exit 1
fi

if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo "Error: Experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

# Validate mode
MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
if [[ ! "$MODE" =~ ^(rgbd|mono)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be: rgbd, mono"
    exit 1
fi

# Convert experiments file to absolute path
EXPERIMENTS_FILE="$(realpath "$EXPERIMENTS_FILE")"

# Paths
BASE_DIR="/media/tam/DATA/3D/CG-photo"

# Set config file based on mode
case "$MODE" in
    rgbd)
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
        MODE_UPPER="RGB-D"
        ;;
    mono)
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml"
        MODE_UPPER="Monocular"
        ;;
esac

CONFIG_BACKUP="${CONFIG_FILE}.backup"
DATASET="tum"

cd "$BASE_DIR"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Backup original config
cp "$CONFIG_FILE" "$CONFIG_BACKUP"
echo "[✓] Config backed up to: $CONFIG_BACKUP"

# Function to restore config
restore_config() {
    if [ -f "$CONFIG_BACKUP" ]; then
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        echo "[✓] Config restored from backup"
    fi
}

# Trap to restore config on exit/error
trap restore_config EXIT

# Function to update config file
update_config() {
    local LAMBDA_DSSIM=$1
    local LAMBDA_GEO=$2
    local LAMBDA_SMOOTH=$3
    local LAMBDA_VAR=$4
    local LAMBDA_ISO=$5
    local LAMBDA_ALIGN=$6
    local FLATTEN_Z=$7
    local LAMBDA_G1=${8:-0.0}
    local LAMBDA_G2=${9:-0.0}
    
    echo "Updating config:"
    echo "  lambda_dssim: $LAMBDA_DSSIM"
    echo "  lambda_geo: $LAMBDA_GEO"
    echo "  lambda_smooth: $LAMBDA_SMOOTH"
    echo "  lambda_var: $LAMBDA_VAR"
    echo "  lambda_iso: $LAMBDA_ISO"
    echo "  lambda_align: $LAMBDA_ALIGN"
    echo "  flatten_z_scale: $FLATTEN_Z"
    echo "  lambda_g1: $LAMBDA_G1"
    echo "  lambda_g2: $LAMBDA_G2"
    
    # Use sed to update the config file (only update if key exists)
    grep -q "^Optimization.lambda_dssim:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_dssim:.*/Optimization.lambda_dssim: $LAMBDA_DSSIM/" "$CONFIG_FILE"
    grep -q "^Optimization.lambda_geo:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_geo:.*/Optimization.lambda_geo: $LAMBDA_GEO/" "$CONFIG_FILE"
    grep -q "^Optimization.lambda_smooth:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_smooth:.*/Optimization.lambda_smooth: $LAMBDA_SMOOTH/" "$CONFIG_FILE"
    grep -q "^Optimization.lambda_var:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_var:.*/Optimization.lambda_var: $LAMBDA_VAR/" "$CONFIG_FILE"
    grep -q "^Optimization.lambda_iso:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_iso:.*/Optimization.lambda_iso: $LAMBDA_ISO/" "$CONFIG_FILE"
    grep -q "^Optimization.lambda_align:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_align:.*/Optimization.lambda_align: $LAMBDA_ALIGN/" "$CONFIG_FILE"
    grep -q "^GaussianMapper.flatten_z_scale:" "$CONFIG_FILE" && sed -i "s/^GaussianMapper.flatten_z_scale:.*/GaussianMapper.flatten_z_scale: $FLATTEN_Z/" "$CONFIG_FILE"
    grep -q "^Optimization.lambda_g1:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_g1:.*/Optimization.lambda_g1: $LAMBDA_G1/" "$CONFIG_FILE"
    grep -q "^Optimization.lambda_g2:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_g2:.*/Optimization.lambda_g2: $LAMBDA_G2/" "$CONFIG_FILE"
}

# Read experiments file and count valid experiments
EXPERIMENTS=()
while IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z LAMBDA_G1 LAMBDA_G2; do
    # Skip empty lines and comments
    [[ -z "$PA_NAME" || "$PA_NAME" =~ ^# ]] && continue
    # Trim whitespace
    PA_NAME=$(echo "$PA_NAME" | tr -d ' ')
    LAMBDA_DSSIM=$(echo "$LAMBDA_DSSIM" | tr -d ' ')
    LAMBDA_GEO=$(echo "$LAMBDA_GEO" | tr -d ' ')
    LAMBDA_SMOOTH=$(echo "$LAMBDA_SMOOTH" | tr -d ' ')
    LAMBDA_VAR=$(echo "$LAMBDA_VAR" | tr -d ' ')
    LAMBDA_ISO=$(echo "$LAMBDA_ISO" | tr -d ' ')
    LAMBDA_ALIGN=$(echo "$LAMBDA_ALIGN" | tr -d ' ')
    FLATTEN_Z=$(echo "$FLATTEN_Z" | tr -d ' ')
    LAMBDA_G1=$(echo "$LAMBDA_G1" | tr -d ' ')
    LAMBDA_G2=$(echo "$LAMBDA_G2" | tr -d ' ')
    # Default values
    LAMBDA_G1=${LAMBDA_G1:-0.0}
    LAMBDA_G2=${LAMBDA_G2:-0.0}
    
    EXPERIMENTS+=("$PA_NAME,$LAMBDA_DSSIM,$LAMBDA_GEO,$LAMBDA_SMOOTH,$LAMBDA_VAR,$LAMBDA_ISO,$LAMBDA_ALIGN,$FLATTEN_Z,$LAMBDA_G1,$LAMBDA_G2")
done < "$EXPERIMENTS_FILE"

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

# Create results directory structure and summary files
RESULTS_BASE="${BASE_DIR}/results/${DATASET}/${MODE}"
mkdir -p "${RESULTS_BASE}"
MULTI_SUMMARY="${RESULTS_BASE}/multi_experiments_summary.csv"

echo ""
echo "=============================================="
echo "  MULTI-EXPERIMENT PIPELINE (TUM Dataset)"
echo "=============================================="
echo "  Mode: ${MODE_UPPER}"
echo "  Experiments file: $EXPERIMENTS_FILE"
echo "  Config file: $CONFIG_FILE"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  Scenes: ${SCENES[*]}"
echo "  Runs per scene: $NUM_RUNS"
echo "  Total runs: $((TOTAL_EXPERIMENTS * ${#SCENES[@]} * NUM_RUNS))"
echo "  Results dir: ${RESULTS_BASE}"
echo "=============================================="
echo ""

echo "experiment,lambda_dssim,lambda_geo,lambda_smooth,lambda_var,lambda_iso,lambda_align,flatten_z,scene,avg_psnr,avg_ssim,avg_lpips,avg_ate,avg_accuracy,avg_completion,avg_comp_ratio" > "$MULTI_SUMMARY"

# Run each experiment
EXP_COUNT=0
for EXP in "${EXPERIMENTS[@]}"; do
    EXP_COUNT=$((EXP_COUNT + 1))
    
    IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z LAMBDA_G1 LAMBDA_G2 <<< "$EXP"
    
    echo ""
    echo "######################################################"
    echo "  EXPERIMENT $EXP_COUNT/$TOTAL_EXPERIMENTS: $PA_NAME"
    echo "  Mode: ${MODE_UPPER}, Dataset: TUM"
    echo "######################################################"
    echo "  Config: dssim=$LAMBDA_DSSIM, geo=$LAMBDA_GEO, var=$LAMBDA_VAR, iso=$LAMBDA_ISO, align=$LAMBDA_ALIGN, g1=$LAMBDA_G1, g2=$LAMBDA_G2"
    echo ""
    
    # Update config file
    update_config "$LAMBDA_DSSIM" "$LAMBDA_GEO" "$LAMBDA_SMOOTH" "$LAMBDA_VAR" "$LAMBDA_ISO" "$LAMBDA_ALIGN" "$FLATTEN_Z" "$LAMBDA_G1" "$LAMBDA_G2"
    
    # Run full pipeline for this experiment
    echo "Running TUM pipeline for $PA_NAME..."
    ./scripts/run_full_pipeline_tum.sh "$MODE" "$PA_NAME" "$NUM_RUNS" "${SCENES[@]}"
    
    # Calculate average metrics
    RESULTS_FILE="${RESULTS_BASE}/${PA_NAME}/all_results.csv"
    if [ -f "$RESULTS_FILE" ]; then
        for SCENE in "${SCENES[@]}"; do
            AVG_PSNR=$(awk -F',' -v scene="$SCENE" '$1==scene && $3!="FAILED" && $3!="N/A" {sum+=$3; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_SSIM=$(awk -F',' -v scene="$SCENE" '$1==scene && $4!="FAILED" && $4!="N/A" {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_LPIPS=$(awk -F',' -v scene="$SCENE" '$1==scene && $5!="FAILED" && $5!="N/A" {sum+=$5; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_ATE=$(awk -F',' -v scene="$SCENE" '$1==scene && $6!="FAILED" && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.6f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_ACC=$(awk -F',' -v scene="$SCENE" '$1==scene && $7!="FAILED" && $7!="N/A" {sum+=$7; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_COMP=$(awk -F',' -v scene="$SCENE" '$1==scene && $8!="FAILED" && $8!="N/A" {sum+=$8; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_RATIO=$(awk -F',' -v scene="$SCENE" '$1==scene && $9!="FAILED" && $9!="N/A" {sum+=$9; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            
            echo "$PA_NAME,$LAMBDA_DSSIM,$LAMBDA_GEO,$LAMBDA_SMOOTH,$LAMBDA_VAR,$LAMBDA_ISO,$LAMBDA_ALIGN,$FLATTEN_Z,$SCENE,$AVG_PSNR,$AVG_SSIM,$AVG_LPIPS,$AVG_ATE,$AVG_ACC,$AVG_COMP,$AVG_RATIO" >> "$MULTI_SUMMARY"
        done
    fi
    
    echo "[✓] Experiment $PA_NAME completed!"
done

# Restore original config
restore_config

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETED! (TUM)"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Summary CSV: $MULTI_SUMMARY"
echo ""
echo "Individual experiment results:"
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r PA_NAME _ <<< "$EXP"
    echo "  - results/${DATASET}/${MODE}/${PA_NAME}/all_results.csv"
done
echo ""
echo "[✓] Multi-experiment TUM pipeline completed successfully!"
