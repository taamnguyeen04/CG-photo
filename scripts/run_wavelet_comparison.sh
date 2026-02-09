#!/bin/bash
# ============================================================================
# WAVELET DENSIFICATION COMPARISON EXPERIMENT
# ============================================================================
# Runs experiments comparing Wavelet Densification vs Baseline (no Wavelet)
# on all Replica rooms for both RGB-D and Monocular modes.
#
# Usage:
#   ./run_wavelet_comparison.sh [num_runs]
#   
#   num_runs: Number of runs per configuration (default: 2)
#
# Output:
#   results/replica/rgbd/wavelet_on/all_results.csv
#   results/replica/rgbd/wavelet_off/all_results.csv
#   results/replica/mono/wavelet_on/all_results.csv
#   results/replica/mono/wavelet_off/all_results.csv
#
# Example:
#   ./run_wavelet_comparison.sh 2    # 2 runs per config = 64 total runs
# ============================================================================

set -e

NUM_RUNS="${1:-1}"
BASE_DIR="/media/tam/DATA/3D/CG-photo"
cd "$BASE_DIR"

# All Replica rooms
SCENES="office0 office1 office2 office3 office4 room0 room1 room2"

# Config files
RGBD_CONFIG="${BASE_DIR}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
MONO_CONFIG="${BASE_DIR}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"

echo "=============================================="
echo "  WAVELET DENSIFICATION COMPARISON"
echo "=============================================="
echo ""
echo "  Runs per config: ${NUM_RUNS}"
echo "  Rooms: ${SCENES}"
echo "  Total experiments: $(echo $SCENES | wc -w) rooms × ${NUM_RUNS} runs × 2 modes × 2 configs = $(($(echo $SCENES | wc -w) * NUM_RUNS * 2 * 2)) runs"
echo ""
echo "  Estimated time: $(($(echo $SCENES | wc -w) * NUM_RUNS * 2 * 2 * 5 / 60)) hours"
echo ""

# Function to set wavelet config
set_wavelet() {
    local CONFIG_FILE=$1
    local ENABLED=$2
    
    if [ "$ENABLED" == "1" ]; then
        sed -i 's/WaveletPyramid.enabled: 0/WaveletPyramid.enabled: 1/' "$CONFIG_FILE"
        echo "[Config] Wavelet Densification: ENABLED"
    else
        sed -i 's/WaveletPyramid.enabled: 1/WaveletPyramid.enabled: 0/' "$CONFIG_FILE"
        echo "[Config] Wavelet Densification: DISABLED"
    fi
}

# ============================================================================
# PART 1: RGB-D MODE
# ============================================================================

echo ""
echo "##############################################"
echo "  PART 1: RGB-D MODE"
echo "##############################################"

# 1a. Wavelet OFF (Baseline)
echo ""
echo ">>> RGB-D: Wavelet OFF (Baseline)"
set_wavelet "$RGBD_CONFIG" 0
./scripts/run_full_pipeline.sh rgbd wavelet_off ${NUM_RUNS} ${SCENES}

# 1b. Wavelet ON
echo ""
echo ">>> RGB-D: Wavelet ON"
set_wavelet "$RGBD_CONFIG" 1
./scripts/run_full_pipeline.sh rgbd wavelet_on ${NUM_RUNS} ${SCENES}

# ============================================================================
# PART 2: MONOCULAR MODE
# ============================================================================

# echo ""
# echo "##############################################"
# echo "  PART 2: MONOCULAR MODE"
# echo "##############################################"

# # 2a. Wavelet OFF (Baseline)
# echo ""
# echo ">>> Mono: Wavelet OFF (Baseline)"
# set_wavelet "$MONO_CONFIG" 0
# ./scripts/run_full_pipeline.sh mono wavelet_off ${NUM_RUNS} ${SCENES}

# # 2b. Wavelet ON
# echo ""
# echo ">>> Mono: Wavelet ON"
# set_wavelet "$MONO_CONFIG" 1
# ./scripts/run_full_pipeline.sh mono wavelet_on ${NUM_RUNS} ${SCENES}

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "##############################################"
echo "  EXPERIMENT COMPLETE!"
echo "##############################################"
echo ""
echo "Results saved to:"
echo "  - results/replica/rgbd/wavelet_off/all_results.csv"
echo "  - results/replica/rgbd/wavelet_on/all_results.csv"
echo "  - results/replica/mono/wavelet_off/all_results.csv"
echo "  - results/replica/mono/wavelet_on/all_results.csv"
echo ""
echo "To compare results:"
echo "  python scripts/compare_wavelet_results.py"
echo ""
