#!/bin/bash
# Ablation study: Run each config on specified scenes
# Usage: ./scripts/run_ablation.sh <num_runs> <scene1> [scene2] ...
# Example: ./scripts/run_ablation.sh 2 office0

set -e

NUM_RUNS="${1:?Usage: ./scripts/run_ablation.sh <num_runs> <scene1> [scene2] ...}"
shift 1
SCENES=("$@")

if [ ${#SCENES[@]} -eq 0 ]; then
    echo "Error: No scenes specified!"
    exit 1
fi

BASE_DIR="/media/tam/DATA/3D/CG-photo"
CONFIG_DIR="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/Replica"
ABLATION_DIR="${CONFIG_DIR}/ablation"
MAIN_CONFIG="${CONFIG_DIR}/replica_rgbd.yaml"

# Ablation configs in order
CONFIGS=("baseline" "conegs" "efd" "mig" "full")

echo "=============================================="
echo "  ABLATION STUDY"
echo "  Scenes: ${SCENES[*]}"
echo "  Runs per config: ${NUM_RUNS}"
echo "  Configs: ${CONFIGS[*]}"
echo "  Total runs: $((${#SCENES[@]} * NUM_RUNS * ${#CONFIGS[@]}))"
echo "=============================================="

# Backup original config
cp "${MAIN_CONFIG}" "${MAIN_CONFIG}.bak"
echo "[✓] Backed up original config to replica_rgbd.yaml.bak"

for CONFIG_NAME in "${CONFIGS[@]}"; do
    CONFIG_FILE="${ABLATION_DIR}/${CONFIG_NAME}.yaml"
    PA_NAME="abl_${CONFIG_NAME}"
    
    echo ""
    echo "######################################################"
    echo "  ABLATION: ${CONFIG_NAME}"
    echo "  Config: ${CONFIG_FILE}"
    echo "  PA Name: ${PA_NAME}"
    echo "######################################################"
    
    # Swap config
    cp "${CONFIG_FILE}" "${MAIN_CONFIG}"
    echo "[✓] Loaded config: ${CONFIG_NAME}"
    
    # Verify what's enabled
    echo "  ConeGS: $(grep '^ConeScale.enabled:' "${MAIN_CONFIG}" | awk '{print $2}')"
    echo "  ConeDensify: $(grep '^ConeDensify.enabled:' "${MAIN_CONFIG}" | awk '{print $2}')"
    echo "  EFD: $(grep '^EFD.enabled:' "${MAIN_CONFIG}" | awk '{print $2}')"
    echo "  MIG: $(grep '^MIG.enabled:' "${MAIN_CONFIG}" | awk '{print $2}')"
    
    # Run pipeline
    cd "${BASE_DIR}"
    ./scripts/run_full_pipeline.sh rgbd "${PA_NAME}" "${NUM_RUNS}" "${SCENES[@]}"
done

# Restore original config
cp "${MAIN_CONFIG}.bak" "${MAIN_CONFIG}"
echo ""
echo "[✓] Restored original config"

# Print comparison table
echo ""
echo "=============================================="
echo "  ABLATION RESULTS COMPARISON"
echo "=============================================="
echo ""

for CONFIG_NAME in "${CONFIGS[@]}"; do
    PA_NAME="abl_${CONFIG_NAME}"
    CSV="${BASE_DIR}/results/replica/rgbd/${PA_NAME}/all_results.csv"
    if [ -f "$CSV" ]; then
        echo "--- ${CONFIG_NAME} ---"
        cat "$CSV" | column -t -s','
        echo ""
    fi
done
