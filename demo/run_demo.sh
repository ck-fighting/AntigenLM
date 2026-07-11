#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_DIR="$ROOT/LLM/AntigenLM"
CLASSIFIER="$ROOT/Downstream/trained_model/protective_antigen/cluster_aware_40_70_15_15_splits/fold_1_seed22_AntigenLM.pt"
INPUT="$ROOT/demo/input/demo_input.csv"
OUTPUT_DIR="$ROOT/demo/output"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: AntigenLM model directory not found: $MODEL_DIR"
    exit 1
fi

if [[ ! -f "$CLASSIFIER" ]]; then
    echo "Error: protective-antigen classifier not found: $CLASSIFIER"
    echo "Please update CLASSIFIER in demo/run_demo.sh to the actual checkpoint path."
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "Error: demo input file not found: $INPUT"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import pandas, torch, transformers" >/dev/null 2>&1; then
    echo "Error: required Python packages are not available from: $PYTHON_BIN"
    echo "Please create and activate the project environment first:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate AntigenLM"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$ROOT/demo/predict.py" \
    --input "$INPUT" \
    --model-dir "$MODEL_DIR" \
    --classifier "$CLASSIFIER" \
    --output-dir "$OUTPUT_DIR"

echo
echo "Demo completed."
echo "Predictions:"
echo "$OUTPUT_DIR/AntigenLM_cluster_aware_fold_1_pred_results.csv"
echo "Metrics:"
echo "$OUTPUT_DIR/AntigenLM_cluster_aware_fold_1_metrics.csv"
