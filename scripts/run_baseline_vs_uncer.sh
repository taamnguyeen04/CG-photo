#!/bin/bash
# ============================================================================
# Compare Baseline vs Uncertainty on TUM dataset
# Runs each config 5 times with 3 runs per experiment (15 total runs each)
# ============================================================================

set -e

BASE_DIR="/media/tam/DATA/3D/CG-photo"
cd "$BASE_DIR"

echo "=============================================="
echo "  BASELINE vs UNCERTAINTY COMPARISON"
echo "  5 experiments × 3 runs = 15 runs each"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Run 5 experiments
for i in {1..5}; do
    echo ""
    echo "######################################################"
    echo "  EXPERIMENT $i/5"
    echo "######################################################"
    echo ""
    
    # Baseline (no uncertainty)
    echo "--- Running BASELINE (tum_baseline_$i) ---"
    ./scripts/run_full_pipeline_tum_all.sh "tum_baseline_$i" 3
    
    echo ""
    
    # Uncertainty enabled
    echo "--- Running UNCERTAINTY (tum_uncer_$i) ---"
    ./scripts/run_full_pipeline_tum_all.sh "tum_uncer_$i" 3 --uncer
    
    echo ""
    echo "[✓] Experiment $i completed!"
    echo ""
done

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETED!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
for i in {1..5}; do
    echo "  - results_tum_baseline_$i/all_results.csv"
    echo "  - results_tum_uncer_$i/all_results.csv"
done
echo ""
echo "[✓] Done! Now analyze results with:"
echo "    python3 scripts/analyze_baseline_vs_uncer.py"
