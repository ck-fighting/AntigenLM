# AntigenLM Protective-antigen Prediction Demo

This demo predicts whether protein sequences are protective antigens using the released AntigenLM encoder and the cluster-aware fold 1 downstream classifier.

## Directory Structure

```text
demo/
├── input/
│   └── demo_input.csv
├── reference/
│   ├── expected_metrics.csv
│   └── expected_predictions.csv
├── output/
│   └── .gitkeep
├── predict.py
├── run_demo.sh
└── README.md
```

- `input/` contains the example sequences and labels.
- `reference/` contains predictions and metrics generated in the tested environment.
- `output/` is recreated locally when the demo is run; generated files are ignored by Git.
- `predict.py` performs embedding extraction, classifier inference, and metric calculation.
- `run_demo.sh` validates all required files and provides the main entry point.

## Input

The example input is `demo/input/demo_input.csv` and contains:

- `ID`: sequence identifier
- `sequence`: protein amino-acid sequence
- `label`: known class label (`1` for a protective antigen and `0` otherwise)

Labels are used only to evaluate the example predictions. The file contains 100 sequences: 42 positive examples and 58 negative examples. Some long proteins are truncated to the model window used by `Downstream/protective_antigen/feature_extractor.py`.

## Required Models

Place the pretrained AntigenLM encoder at:

```text
LLM/AntigenLM/
```

The classifier checkpoint is:

```text
Downstream/trained_model/protective_antigen/30_similarity/fold_1_seed22_AntigenLM.pt
```

## Run

Create and activate the project environment if needed:

```bash
conda env create -f environment.yml
conda activate AntigenLM
```

Then run from the repository root:

```bash
bash demo/run_demo.sh
```

To select a particular Python executable:

```bash
PYTHON_BIN=/path/to/python bash demo/run_demo.sh
```

## Run on Your Own Data

Prepare a CSV file containing the `ID`, `sequence`, and `label` columns, where `label` is `1` for a protective antigen and `0` otherwise. From the repository root, replace `/path/to/your_data.csv` with your input file and run:

```bash
conda activate AntigenLM
python demo/predict.py \
  --input /path/to/your_data.csv \
  --model-dir LLM/AntigenLM \
  --classifier Downstream/trained_model/protective_antigen/30_similarity/fold_1_seed22_AntigenLM.pt \
  --output-dir demo/output/your_data
```

Predictions and evaluation metrics will be written to `demo/output/your_data/`. Use `--threshold`, `--batch-size`, or `--extract-batch-size` to override their default values when needed.

## Output

The demo writes:

```text
demo/output/AntigenLM_cluster_aware_fold_1_pred_results.csv
demo/output/AntigenLM_cluster_aware_fold_1_metrics.csv
```

The prediction columns are `id`, `y_true`, `y_pred`, and `y_score`. Reference files are provided at `demo/reference/expected_predictions.csv` and `demo/reference/expected_metrics.csv`. Minor numerical differences may occur across GPU and CUDA versions.

## Tested Environment

- Ubuntu 20.04.6 LTS
- Python 3.10.20
- PyTorch 2.2.2+cu121
- Transformers 4.45.2
- CUDA runtime 12.1
- NVIDIA A100-SXM4-80GB

## Expected Runtime

On the tested Ubuntu 20.04.6 LTS system with an NVIDIA A100-SXM4-80GB GPU, the demo processes 100 protein sequences in approximately 0.1 minutes (about 6 seconds). This runtime includes model loading, embedding extraction, and classifier inference, but excludes model and checkpoint download time.

CPU-only execution has not been benchmarked and is not recommended for this demo.
