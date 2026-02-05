#!/bin/bash
# ============================================================================
# Multi-Experiment Pipeline for Photo-SLAM
# ============================================================================
# Run multiple experiments with different loss configurations automatically.
# Perfect for overnight/weekend experiments without manual intervention.
#
# Usage:
#   ./run_multi_experiments.sh <experiments_file> <num_runs> <scene1> [scene2] ...
#
# Examples:
#   ./run_multi_experiments.sh experiments/example_experiments.txt 1 office0 room0
#   ./run_multi_experiments.sh experiments/example_experiments.txt 1 office0 office1 office2 office3 office4 room0 room1 room2
#
# experiments_file format (CSV):
#   # Comment lines start with #
#   # PA_NAME,lambda_dssim,lambda_geo,lambda_smooth,lambda_var,lambda_iso,lambda_align,flatten_z
#   pa13,0.2,0.5,0.0,0.0,0.0,0.0,false
#   pa14,0.2,0.8,0.01,0.0,0.0,0.0,true
# ============================================================================

set -e

# Parse arguments
EXPERIMENTS_FILE="${1:?Usage: ./run_multi_experiments.sh <experiments_file> <num_runs> <scene1> [scene2] ...}"
NUM_RUNS="${2:?Missing number of runs}"
shift 2
SCENES=("$@")

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No scenes specified!"
    echo "Usage: ./run_multi_experiments.sh <experiments_file> <num_runs> <scene1> [scene2] ..."
    exit 1
fi

if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo "Error: Experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

# Convert experiments file to absolute path before cd
EXPERIMENTS_FILE="$(realpath "$EXPERIMENTS_FILE")"

# Paths
BASE_DIR="/media/tam/DATA/3D/CG-photo"
CONFIG_FILE="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
CONFIG_BACKUP="${CONFIG_FILE}.backup"

cd "$BASE_DIR"

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
    
    echo "Updating config:"
    echo "  lambda_dssim: $LAMBDA_DSSIM"
    echo "  lambda_geo: $LAMBDA_GEO"
    echo "  lambda_smooth: $LAMBDA_SMOOTH"
    echo "  lambda_var: $LAMBDA_VAR"
    echo "  lambda_iso: $LAMBDA_ISO"
    echo "  lambda_align: $LAMBDA_ALIGN"
    echo "  flatten_z_scale: $FLATTEN_Z"
    
    # Use sed to update the config file
    sed -i "s/^Optimization.lambda_dssim:.*/Optimization.lambda_dssim: $LAMBDA_DSSIM/" "$CONFIG_FILE"
    sed -i "s/^Optimization.lambda_geo:.*/Optimization.lambda_geo: $LAMBDA_GEO/" "$CONFIG_FILE"
    sed -i "s/^Optimization.lambda_smooth:.*/Optimization.lambda_smooth: $LAMBDA_SMOOTH/" "$CONFIG_FILE"
    sed -i "s/^Optimization.lambda_var:.*/Optimization.lambda_var: $LAMBDA_VAR/" "$CONFIG_FILE"
    sed -i "s/^Optimization.lambda_iso:.*/Optimization.lambda_iso: $LAMBDA_ISO/" "$CONFIG_FILE"
    sed -i "s/^Optimization.lambda_align:.*/Optimization.lambda_align: $LAMBDA_ALIGN/" "$CONFIG_FILE"
    sed -i "s/^GaussianMapper.flatten_z_scale:.*/GaussianMapper.flatten_z_scale: $FLATTEN_Z/" "$CONFIG_FILE"
}

# Read experiments file and count valid experiments
EXPERIMENTS=()
while IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z; do
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
    
    EXPERIMENTS+=("$PA_NAME,$LAMBDA_DSSIM,$LAMBDA_GEO,$LAMBDA_SMOOTH,$LAMBDA_VAR,$LAMBDA_ISO,$LAMBDA_ALIGN,$FLATTEN_Z")
done < "$EXPERIMENTS_FILE"

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}

echo ""
echo "=============================================="
echo "  MULTI-EXPERIMENT PIPELINE"
echo "=============================================="
echo "  Experiments file: $EXPERIMENTS_FILE"
echo "  Total experiments: $TOTAL_EXPERIMENTS"
echo "  Scenes: ${SCENES[*]}"
echo "  Runs per scene: $NUM_RUNS"
echo "  Total runs: $((TOTAL_EXPERIMENTS * ${#SCENES[@]} * NUM_RUNS))"
echo "=============================================="
echo ""

# Create summary file for all experiments
MULTI_SUMMARY="${BASE_DIR}/multi_experiments_summary.csv"
echo "experiment,lambda_dssim,lambda_geo,lambda_smooth,lambda_var,lambda_iso,lambda_align,flatten_z,scene,avg_psnr,avg_ssim,avg_lpips,avg_ate,avg_accuracy,avg_completion,avg_comp_ratio" > "$MULTI_SUMMARY"

# Run each experiment
EXP_COUNT=0
for EXP in "${EXPERIMENTS[@]}"; do
    EXP_COUNT=$((EXP_COUNT + 1))
    
    IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z <<< "$EXP"
    
    echo ""
    echo "######################################################"
    echo "  EXPERIMENT $EXP_COUNT/$TOTAL_EXPERIMENTS: $PA_NAME"
    echo "######################################################"
    echo "  Config: dssim=$LAMBDA_DSSIM, geo=$LAMBDA_GEO, smooth=$LAMBDA_SMOOTH, var=$LAMBDA_VAR, iso=$LAMBDA_ISO, align=$LAMBDA_ALIGN, flatten=$FLATTEN_Z"
    echo ""
    
    # Update config file
    update_config "$LAMBDA_DSSIM" "$LAMBDA_GEO" "$LAMBDA_SMOOTH" "$LAMBDA_VAR" "$LAMBDA_ISO" "$LAMBDA_ALIGN" "$FLATTEN_Z"
    
    # Run full pipeline for this experiment
    echo "Running full pipeline for $PA_NAME..."
    ./scripts/run_full_pipeline.sh "$PA_NAME" "$NUM_RUNS" "${SCENES[@]}"
    
    # Calculate average metrics for this experiment
    RESULTS_FILE="${BASE_DIR}/results_${PA_NAME}/all_results.csv"
    if [ -f "$RESULTS_FILE" ]; then
        # Calculate averages per scene
        # Calculate averages per scene (column indices: 3=psnr, 4=ssim, 5=lpips, 6=ate, 7=acc, 8=comp, 9=ratio)
        for SCENE in "${SCENES[@]}"; do
            AVG_PSNR=$(awk -F',' -v scene="$SCENE" '$1==scene {sum+=$3; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_SSIM=$(awk -F',' -v scene="$SCENE" '$1==scene {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_LPIPS=$(awk -F',' -v scene="$SCENE" '$1==scene {sum+=$5; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_ATE=$(awk -F',' -v scene="$SCENE" '$1==scene && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.6f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_ACC=$(awk -F',' -v scene="$SCENE" '$1==scene {sum+=$7; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_COMP=$(awk -F',' -v scene="$SCENE" '$1==scene {sum+=$8; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_RATIO=$(awk -F',' -v scene="$SCENE" '$1==scene {sum+=$9; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            
            echo "$PA_NAME,$LAMBDA_DSSIM,$LAMBDA_GEO,$LAMBDA_SMOOTH,$LAMBDA_VAR,$LAMBDA_ISO,$LAMBDA_ALIGN,$FLATTEN_Z,$SCENE,$AVG_PSNR,$AVG_SSIM,$AVG_LPIPS,$AVG_ATE,$AVG_ACC,$AVG_COMP,$AVG_RATIO" >> "$MULTI_SUMMARY"
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

REPORT_FILE="${BASE_DIR}/multi_experiments_report.md"

cat > "$REPORT_FILE" << 'HEADER'
# Multi-Experiment Comparison Report

Generated: $(date)

## Experiment Configurations

| Experiment | λ_dssim | λ_geo | λ_smooth | λ_var | λ_iso | λ_align | Flatten Z |
|------------|---------|-------|----------|-------|-------|---------|-----------|
HEADER

# Add experiment configs to report
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z <<< "$EXP"
    echo "| $PA_NAME | $LAMBDA_DSSIM | $LAMBDA_GEO | $LAMBDA_SMOOTH | $LAMBDA_VAR | $LAMBDA_ISO | $LAMBDA_ALIGN | $FLATTEN_Z |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << 'SECTION'

## Overall Results (Average Across All Scenes)

| Experiment | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ATE ↓ | Accuracy ↓ | Completion ↓ | Comp.Ratio ↑ |
|------------|--------|--------|---------|-------|------------|--------------|--------------|
SECTION

# Calculate overall averages for each experiment
for EXP in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z <<< "$EXP"
    
    RESULTS_FILE="${BASE_DIR}/results_${PA_NAME}/all_results.csv"
    if [ -f "$RESULTS_FILE" ]; then
        AVG_PSNR=$(awk -F',' 'NR>1 && $3!="FAILED" {sum+=$3; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_SSIM=$(awk -F',' 'NR>1 && $4!="FAILED" {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_LPIPS=$(awk -F',' 'NR>1 && $5!="FAILED" {sum+=$5; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_ATE=$(awk -F',' 'NR>1 && $6!="FAILED" && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.6f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_ACC=$(awk -F',' 'NR>1 && $7!="FAILED" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_COMP=$(awk -F',' 'NR>1 && $8!="FAILED" {sum+=$8; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        AVG_RATIO=$(awk -F',' 'NR>1 && $9!="FAILED" {sum+=$9; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
        
        echo "| **$PA_NAME** | $AVG_PSNR | $AVG_SSIM | $AVG_LPIPS | $AVG_ATE | $AVG_ACC | $AVG_COMP | $AVG_RATIO |" >> "$REPORT_FILE"
    fi
done

# Add per-scene comparison
cat >> "$REPORT_FILE" << 'SECTION'

## Per-Scene Comparison

SECTION

for SCENE in "${SCENES[@]}"; do
    echo "### $SCENE" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| Experiment | PSNR | SSIM | LPIPS | ATE | Accuracy | Completion | Comp.Ratio |" >> "$REPORT_FILE"
    echo "|------------|------|------|-------|-----|----------|------------|------------|" >> "$REPORT_FILE"
    
    for EXP in "${EXPERIMENTS[@]}"; do
        IFS=',' read -r PA_NAME LAMBDA_DSSIM LAMBDA_GEO LAMBDA_SMOOTH LAMBDA_VAR LAMBDA_ISO LAMBDA_ALIGN FLATTEN_Z <<< "$EXP"
        
        RESULTS_FILE="${BASE_DIR}/results_${PA_NAME}/all_results.csv"
        if [ -f "$RESULTS_FILE" ]; then
            AVG_PSNR=$(awk -F',' -v scene="$SCENE" '$1==scene && $3!="FAILED" {sum+=$3; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_SSIM=$(awk -F',' -v scene="$SCENE" '$1==scene && $4!="FAILED" {sum+=$4; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_LPIPS=$(awk -F',' -v scene="$SCENE" '$1==scene && $5!="FAILED" {sum+=$5; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_ATE=$(awk -F',' -v scene="$SCENE" '$1==scene && $6!="FAILED" && $6!="N/A" {sum+=$6; count++} END {if(count>0) printf "%.6f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_ACC=$(awk -F',' -v scene="$SCENE" '$1==scene && $7!="FAILED" {sum+=$7; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_COMP=$(awk -F',' -v scene="$SCENE" '$1==scene && $8!="FAILED" {sum+=$8; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            AVG_RATIO=$(awk -F',' -v scene="$SCENE" '$1==scene && $9!="FAILED" {sum+=$9; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "$RESULTS_FILE")
            
            echo "| $PA_NAME | $AVG_PSNR | $AVG_SSIM | $AVG_LPIPS | $AVG_ATE | $AVG_ACC | $AVG_COMP | $AVG_RATIO |" >> "$REPORT_FILE"
        fi
    done
    echo "" >> "$REPORT_FILE"
done

# Find best experiment
cat >> "$REPORT_FILE" << 'SECTION'

## Best Configurations

SECTION

# Find best for each metric
echo "| Metric | Best Experiment | Value |" >> "$REPORT_FILE"
echo "|--------|-----------------|-------|" >> "$REPORT_FILE"

# This would require more complex logic, so we'll add a placeholder
echo "_Best configurations analysis will be added in future versions._" >> "$REPORT_FILE"

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
