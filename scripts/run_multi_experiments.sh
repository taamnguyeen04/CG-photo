#!/bin/bash
# ============================================================================
# Multi-Experiment Pipeline for Photo-SLAM on Replica
# ============================================================================
# Run multiple experiments with different loss configurations automatically.
# 
# Supports: RGB-D, Monocular, Stereo modes
#
# Usage:
#   ./run_multi_experiments.sh <mode> <experiments_file> <num_runs> <scene1> [scene2] ...
#
# Modes: rgbd, mono, stereo
#
# Examples:
#   ./run_multi_experiments.sh rgbd experiments.txt 2 office0 room0
#   ./run_multi_experiments.sh mono experiments.txt 1 office0 office1
# ============================================================================

set -e

# Parse arguments
MODE="${1:?Usage: ./run_multi_experiments.sh <mode> <experiments_file> <num_runs> <scene1> [scene2] ...}"
EXPERIMENTS_FILE="${2:?Missing experiments file}"
NUM_RUNS="${3:?Missing number of runs}"
shift 3
SCENES=("$@")

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No scenes specified!"
    echo "Usage: ./run_multi_experiments.sh <mode> <experiments_file> <num_runs> <scene1> [scene2] ..."
    exit 1
fi

if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo "Error: Experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

# Validate mode
MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
if [[ ! "$MODE" =~ ^(rgbd|mono|stereo)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be: rgbd, mono, stereo"
    exit 1
fi

# Convert experiments file to absolute path before cd
EXPERIMENTS_FILE="$(realpath "$EXPERIMENTS_FILE")"

# Paths
BASE_DIR="/media/tam/DATA/3D/CG-photo"

# Set config file based on mode
case "$MODE" in
    rgbd)
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
        MODE_UPPER="RGB-D"
        ;;
    mono)
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
        MODE_UPPER="Monocular"
        ;;
    stereo)
        CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/Stereo/Replica/replica_stereo.yaml"
        MODE_UPPER="Stereo"
        ;;
esac

CONFIG_BACKUP="${CONFIG_FILE}.backup"
DATASET="replica"

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
    local LAMBDA_REG=${10:-0.0}
    
    # Force lambda_geo=0 and lambda_var=0 for mono/stereo (no GT depth available)
    if [[ "$MODE" == "mono" || "$MODE" == "stereo" ]]; then
        if [[ "$LAMBDA_GEO" != "0.0" && "$LAMBDA_GEO" != "0" ]]; then
            echo "[!] WARNING: Forcing lambda_geo=0 for $MODE mode (no GT depth)"
            LAMBDA_GEO="0.0"
        fi
        if [[ "$LAMBDA_VAR" != "0.0" && "$LAMBDA_VAR" != "0" ]]; then
            echo "[!] WARNING: Forcing lambda_var=0 for $MODE mode (no GT depth)"
            LAMBDA_VAR="0.0"
        fi
    fi
    
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
    echo "  lambda_reg: $LAMBDA_REG"
    
    # Use sed to update the config file (only update if key exists)
    # Note: || true prevents set -e from stopping on grep not finding key
    (grep -q "^Optimization.lambda_dssim:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_dssim:.*/Optimization.lambda_dssim: $LAMBDA_DSSIM/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_geo:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_geo:.*/Optimization.lambda_geo: $LAMBDA_GEO/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_smooth:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_smooth:.*/Optimization.lambda_smooth: $LAMBDA_SMOOTH/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_var:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_var:.*/Optimization.lambda_var: $LAMBDA_VAR/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_iso:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_iso:.*/Optimization.lambda_iso: $LAMBDA_ISO/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_align:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_align:.*/Optimization.lambda_align: $LAMBDA_ALIGN/" "$CONFIG_FILE") || true
    (grep -q "^GaussianMapper.flatten_z_scale:" "$CONFIG_FILE" && sed -i "s/^GaussianMapper.flatten_z_scale:.*/GaussianMapper.flatten_z_scale: $FLATTEN_Z/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_g1:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_g1:.*/Optimization.lambda_g1: $LAMBDA_G1/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_g2:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_g2:.*/Optimization.lambda_g2: $LAMBDA_G2/" "$CONFIG_FILE") || true
    (grep -q "^Optimization.lambda_reg:" "$CONFIG_FILE" && sed -i "s/^Optimization.lambda_reg:.*/Optimization.lambda_reg: $LAMBDA_REG/" "$CONFIG_FILE") || true
}

# Read experiments file and count valid experiments
# Format: name,dssim,geo,smooth,var,iso,align,flatten,g1,g2,reg
EXPERIMENTS=()
while IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z LAMBDA_G1 LAMBDA_G2 LAMBDA_REG; do
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
    LAMBDA_REG=$(echo "$LAMBDA_REG" | tr -d ' ')
    # Default values for backward compatibility
    LAMBDA_G1=${LAMBDA_G1:-0.0}
    LAMBDA_G2=${LAMBDA_G2:-0.0}
    LAMBDA_REG=${LAMBDA_REG:-0.0}
    
    EXPERIMENTS+=("$PA_NAME,$LAMBDA_DSSIM,$LAMBDA_GEO,$LAMBDA_SMOOTH,$LAMBDA_VAR,$LAMBDA_ISO,$LAMBDA_ALIGN,$FLATTEN_Z,$LAMBDA_G1,$LAMBDA_G2,$LAMBDA_REG")
done < "$EXPERIMENTS_FILE"

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

echo ""
echo "=============================================="
echo "  MULTI-EXPERIMENT PIPELINE"
echo "=============================================="
echo "  Mode: ${MODE_UPPER}"
echo "  Experiments file: $EXPERIMENTS_FILE"
echo "  Config file: $CONFIG_FILE"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  Scenes: ${SCENES[*]}"
echo "  Runs per scene: $NUM_RUNS"
echo "  Total runs: $((TOTAL_EXPERIMENTS * ${#SCENES[@]} * NUM_RUNS))"
echo "=============================================="
echo ""

# Create results directory structure and summary files
RESULTS_BASE="${BASE_DIR}/results/${DATASET}/${MODE}"
mkdir -p "${RESULTS_BASE}"
MULTI_SUMMARY="${RESULTS_BASE}/multi_experiments_summary.csv"
echo "experiment,lambda_dssim,lambda_geo,lambda_smooth,lambda_var,lambda_iso,lambda_align,flatten_z,scene,avg_psnr,avg_ssim,avg_lpips,avg_accuracy,avg_completion,avg_comp_ratio" > "$MULTI_SUMMARY"

# Run each experiment
EXP_COUNT=0
for EXP in "${EXPERIMENTS[@]}"; do
    EXP_COUNT=$((EXP_COUNT + 1))
    
    IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z LAMBDA_G1 LAMBDA_G2 LAMBDA_REG <<< "$EXP"
    
    echo ""
    echo "######################################################"
    echo "  EXPERIMENT $EXP_COUNT/$TOTAL_EXPERIMENTS: $PA_NAME"
    echo "  Mode: ${MODE_UPPER}"
    echo "######################################################"
    echo "  Config: dssim=$LAMBDA_DSSIM, geo=$LAMBDA_GEO, smooth=$LAMBDA_SMOOTH, var=$LAMBDA_VAR, iso=$LAMBDA_ISO, align=$LAMBDA_ALIGN, flatten=$FLATTEN_Z, g1=$LAMBDA_G1, g2=$LAMBDA_G2, reg=$LAMBDA_REG"
    echo ""
    
    # Update config file
    update_config "$LAMBDA_DSSIM" "$LAMBDA_GEO" "$LAMBDA_SMOOTH" "$LAMBDA_VAR" "$LAMBDA_ISO" "$LAMBDA_ALIGN" "$FLATTEN_Z" "$LAMBDA_G1" "$LAMBDA_G2" "$LAMBDA_REG"
    
    # Run full pipeline for this experiment
    echo "Running full pipeline for $PA_NAME..."
    ./scripts/run_full_pipeline.sh "$MODE" "$PA_NAME" "$NUM_RUNS" "${SCENES[@]}"
    
    # Calculate average metrics for this experiment
    RESULTS_FILE="${RESULTS_BASE}/${PA_NAME}/all_results.csv"
    if [ -f "$RESULTS_FILE" ]; then
        # Calculate averages per scene (column indices: 3=psnr, 4=ssim, 5=lpips, 7=acc, 8=comp, 9=ratio)
        for SCENE in "${SCENES[@]}"; do
            AVG_PSNR=$(awk -F',' -v scene="$SCENE" '$1==scene && $3!="FAILED" {sum+=$3; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_SSIM=$(awk -F',' -v scene="$SCENE" '$1==scene && $4!="FAILED" {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_LPIPS=$(awk -F',' -v scene="$SCENE" '$1==scene && $5!="FAILED" {sum+=$5; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_ACC=$(awk -F',' -v scene="$SCENE" '$1==scene && $7!="FAILED" {sum+=$7; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_COMP=$(awk -F',' -v scene="$SCENE" '$1==scene && $8!="FAILED" {sum+=$8; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_RATIO=$(awk -F',' -v scene="$SCENE" '$1==scene && $9!="FAILED" {sum+=$9; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            
            echo "$PA_NAME,$LAMBDA_DSSIM,$LAMBDA_GEO,$LAMBDA_SMOOTH,$LAMBDA_VAR,$LAMBDA_ISO,$LAMBDA_ALIGN,$FLATTEN_Z,$SCENE,$AVG_PSNR,$AVG_SSIM,$AVG_LPIPS,$AVG_ACC,$AVG_COMP,$AVG_RATIO" >> "$MULTI_SUMMARY"
        done
    fi
    
    echo "[✓] Experiment $PA_NAME completed!"
done

# Restore original config
restore_config

# ============================================================================
# Generate Comparison Report
# ============================================================================
echo ""
echo "=============================================="
echo "  GENERATING COMPARISON REPORT"
echo "=============================================="

REPORT_FILE="${RESULTS_BASE}/multi_experiments_report.md"

cat > "$REPORT_FILE" << HEADER
# Multi-Experiment Comparison Report

**Mode**: ${MODE_UPPER}
**Generated**: $(date)

## Experiment Configurations

| Experiment | λ_dssim | λ_geo | λ_smooth | λ_var | λ_iso | λ_align | Flatten Z | λ_g1 | λ_g2 |
|------------|---------|-------|----------|-------|-------|---------|-----------|------|------|
HEADER

# Add experiment configs to report
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z LAMBDA_G1 LAMBDA_G2 <<< "$EXP"
    echo "| $PA_NAME | $LAMBDA_DSSIM | $LAMBDA_GEO | $LAMBDA_SMOOTH | $LAMBDA_VAR | $LAMBDA_ISO | $LAMBDA_ALIGN | $FLATTEN_Z | $LAMBDA_G1 | $LAMBDA_G2 |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << 'SECTION'

## Overall Results (Average Across All Scenes)

| Experiment | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Accuracy ↓ | Completion ↓ | Comp.Ratio ↑ |
|------------|--------|--------|---------|------------|--------------|--------------|
SECTION

# Calculate overall averages for each experiment
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z LAMBDA_G1 LAMBDA_G2 <<< "$EXP"
    
    RESULTS_FILE="${RESULTS_BASE}/${PA_NAME}/all_results.csv"
    if [ -f "$RESULTS_FILE" ]; then
        AVG_PSNR=$(awk -F',' 'NR>1 && $3!="FAILED" {sum+=$3; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_SSIM=$(awk -F',' 'NR>1 && $4!="FAILED" {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_LPIPS=$(awk -F',' 'NR>1 && $5!="FAILED" {sum+=$5; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_ACC=$(awk -F',' 'NR>1 && $7!="FAILED" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_COMP=$(awk -F',' 'NR>1 && $8!="FAILED" {sum+=$8; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_RATIO=$(awk -F',' 'NR>1 && $9!="FAILED" {sum+=$9; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        
        echo "| **$PA_NAME** | $AVG_PSNR | $AVG_SSIM | $AVG_LPIPS | $AVG_ACC | $AVG_COMP | $AVG_RATIO |" >> "$REPORT_FILE"
    fi
done

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETED!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Summary CSV: $MULTI_SUMMARY"
echo "  - Report: $REPORT_FILE"
echo ""
echo "Individual experiment results:"
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r PA_NAME _ <<< "$EXP"
    echo "  - results_${PA_NAME}/all_results.csv"
done
echo ""
echo "[✓] Multi-experiment pipeline completed successfully!"
