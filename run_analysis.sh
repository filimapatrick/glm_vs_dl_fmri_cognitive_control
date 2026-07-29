#!/usr/bin/env bash
# ==============================================================================
# run_analysis.sh - End-to-End Analysis Pipeline Runner
# A Methodological Study of GLM and Deep Learning Behavior in fMRI
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Project root directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python3"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "🧠 GLM vs Deep Learning fMRI Study - End-to-End Pipeline Execution"
echo "Project Root: ${PROJECT_DIR}"
echo "Python Executable: ${PYTHON_BIN}"
echo "======================================================================"

# Stage 1: Data Quality Control & Outlier Exclusion
echo ""
echo "----------------------------------------------------------------------"
echo "🚀 [Stage 1/6] Data Quality Control & Exclusion Criteria"
echo "----------------------------------------------------------------------"
"${PYTHON_BIN}" scripts/01_quality_control.py

# Stage 2: FSL Preprocessing
echo ""
echo "----------------------------------------------------------------------"
echo "🚀 [Stage 2/6] FSL Preprocessing Pipeline"
echo "----------------------------------------------------------------------"
"${PYTHON_BIN}" scripts/02_preprocessing.py

# Stage 3: Classical GLM Analysis
echo ""
echo "----------------------------------------------------------------------"
echo "🚀 [Stage 3/6] Classical GLM First & Group Level Analysis"
echo "----------------------------------------------------------------------"
"${PYTHON_BIN}" scripts/03_glm_analysis.py

# Stage 4: Feature Extraction for Machine Learning
echo ""
echo "----------------------------------------------------------------------"
echo "🚀 [Stage 4/6] Feature Extraction & LOSO CV Partitions"
echo "----------------------------------------------------------------------"
"${PYTHON_BIN}" scripts/04_feature_extraction.py

# Stage 5: Deep Learning & Baseline Classifier Benchmarking
echo ""
echo "----------------------------------------------------------------------"
echo "🚀 [Stage 5/6] Deep Learning & Baseline Classifier Benchmarking"
echo "----------------------------------------------------------------------"
"${PYTHON_BIN}" scripts/05_deep_learning.py

# Stage 6: Comparative & Spatial Attribution Evaluation
echo ""
echo "----------------------------------------------------------------------"
echo "🚀 [Stage 6/6] Comparative & Spatial Attribution Evaluation"
echo "----------------------------------------------------------------------"
"${PYTHON_BIN}" scripts/06_comparison_analysis.py

echo ""
echo "======================================================================"
echo "🎉 ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!"
echo "Results summary: ${PROJECT_DIR}/results/final_study_results.json"
echo "Performance plot: ${PROJECT_DIR}/results/model_performance_comparison.png"
echo "======================================================================"
