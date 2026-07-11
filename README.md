# AntigenLM Codebase Overview

AntigenLM is a protein language model for antigen representation and immune-related prediction. This repository provides the three-stage pre-training pipeline, five downstream benchmarks, a small protective-antigen demo, released results, and figure-generation scripts.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ck-fighting/AntigenLM.git
   cd AntigenLM
   ```

2. Create and activate the Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate AntigenLM
   ```

3. Download the required model files:

   - For downstream prediction and the demo, download [AntigenLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/AntigenLM) to `LLM/AntigenLM/`.
   - To reproduce the complete pre-training pipeline, also download [MicroLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/MicroLM) and [PathogLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/PathogLM) to `LLM/MicroLM/` and `LLM/PathogLM/`, respectively.
   - To evaluate with released downstream classifiers, download the [downstream checkpoints](https://huggingface.co/cckai2017/Downstream_trained_model/tree/main) and retain their directory structure under `Downstream/trained_model/`.

### Installation Time in the Tested Environment

Creating the Conda environment takes approximately 12 minutes on a server running Ubuntu 20.04.6 LTS with two AMD EPYC 7543 32-core processors (64 physical cores in total) and 2 TiB of memory. This estimate excludes the time required to download the pretrained model and downstream classifier checkpoints.

## System Requirements

### Software

- Linux (Ubuntu recommended)
- Conda or Miniconda
- Python 3.10
- PyTorch 2.2.2
- Transformers 4.45.2
- DeepSpeed 0.14.4 for pre-training
- The remaining Python dependencies in `environment.yml`

### Hardware

- An NVIDIA CUDA-compatible GPU is strongly recommended for embedding extraction, training, and inference.
- The pre-training commands below are configured for four GPUs by default. Change `--num_gpus` to match the available hardware.
- The downstream tasks are configured to use one GPU by default.

All experiments were conducted on Ubuntu 20.04.6 LTS with Python 3.10.20, PyTorch 2.2.2+cu121, CUDA 12.1, and NVIDIA A100-SXM4-80GB GPUs.

## Overview

The main repository directories are:

```text
AntigenLM/
├── Pre-training/   # MicroLM -> PathogLM -> AntigenLM training pipeline
├── Downstream/     # Downstream datasets, training, and evaluation code
│   └── Result/   # Figure-ready predictions and summary tables
├── LLM/            # Local pretrained model files
├── demo/           # Small protective-antigen prediction example

```

The model is developed in three stages:

1. **MicroLM:** pre-training on microbial protein sequences.
2. **PathogLM:** fine-tuning MicroLM on pathogen protein sequences and secondary-structure labels.
3. **AntigenLM:** fine-tuning PathogLM on antigen sequences and secondary-structure labels.

The final AntigenLM encoder is evaluated on protective-antigen classification, pHLA-I binding, pHLA-II binding, pTCR recognition, and B-cell epitope prediction.

> [!IMPORTANT]
> Before pre-training, download the datasets described below to `Pre-training/dataset/`. Before running downstream tasks, place the AntigenLM model in `LLM/AntigenLM/`. Evaluation with released classifiers additionally requires the appropriate checkpoint files in `Downstream/trained_model/`.

## 1) Pre-training

### 1.1 Code Structure

- `Pre-training/esmc_pretrain_main.py`: MicroLM pre-training with DeepSpeed.
- `Pre-training/bert_finetuning_esmc_style_last8_layers.py`: fine-tuning script for the PathogLM and AntigenLM stages.
- `Pre-training/data/`: dataset loaders and collators.
- `Pre-training/bert_data_prepare/`: tokenizer and data-preparation utilities.
- `Pre-training/config/`: training configurations.
- `Pre-training/dataset/`: local directory for downloaded training datasets.

### 1.2 Training Datasets

Download the following files from [the AntigenLM dataset repository](https://huggingface.co/datasets/cckai2017/AntigenLM/tree/main) and place them in `Pre-training/dataset/`:

- [`dataset_micro.fasta`](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/dataset_micro.fasta.fasta): raw microbial protein sequences for MicroLM pre-training.
- [`pathogen_seq_ss.csv`](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/pathogen_seq_ss.csv.csv): pathogen sequences and secondary-structure labels for PathogLM fine-tuning.
- [`antigen_seq_ss.csv`](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/antigen_seq_ss_2.csv.csv): antigen sequences and secondary-structure labels for AntigenLM fine-tuning.

The fine-tuning CSV files contain the `sequence` and `second_structure` columns.

### 1.3 Configurations

- `Pre-training/config/bert_pretrain_esmc_300m.json`: MicroLM pre-training.
- `Pre-training/config/bert_finetune_MicroLM_last8_layers.json`: MicroLM to PathogLM fine-tuning.
- `Pre-training/config/bert_finetune_PathogLM_last8_layers.json`: PathogLM to AntigenLM fine-tuning.

Review the dataset, pretrained-model, and output-checkpoint paths in each JSON file before launching a job.

### 1.4 Run Commands

Pre-train MicroLM:

```bash
cd Pre-training
deepspeed --num_gpus=4 esmc_pretrain_main.py \
  -c config/bert_pretrain_esmc_300m.json
```

Fine-tune MicroLM to PathogLM:

```bash
cd Pre-training
deepspeed --num_gpus=4 bert_finetuning_esmc_style_last8_layers.py \
  -c config/bert_finetune_MicroLM_last8_layers.json
```

Fine-tune PathogLM to AntigenLM:

```bash
cd Pre-training
deepspeed --num_gpus=4 bert_finetuning_esmc_style_last8_layers.py \
  -c config/bert_finetune_PathogLM_last8_layers.json
```

Run each command from a fresh repository-root shell, or remain in `Pre-training/` between stages rather than repeating `cd Pre-training`.

## 2) Downstream

The downstream scripts use pretrained embeddings with task-specific prediction heads. Released task checkpoints are available from the [Downstream trained model repository](https://huggingface.co/cckai2017/Downstream_trained_model/tree/main).

### 2.1 Protective Antigen Classification

Train and evaluate five-fold cross-validation:

```bash
cd Downstream/protective_antigen
python protective_antigen_train.py
python protective_antigen_test.py
```

Train on and evaluate the independent bacterial or viral datasets:

```bash
cd Downstream/protective_antigen
python protective_antigen_train.py --mode Independent --subset Bacteria
python protective_antigen_train.py --mode Independent --subset Viruses
python protective_antigen_test.py --mode independent --subset Bacteria
python protective_antigen_test.py --mode independent --subset Viruses
```

### 2.2 pHLA-I Binding

Train and evaluate five-fold cross-validation:

```bash
cd Downstream/pMHC-I
torchrun --standalone --nproc_per_node=1 MHC_train.py --mode cv
python MHC_test.py --mode cv
```

Train and evaluate the independent dataset:

```bash
cd Downstream/pMHC-I
torchrun --standalone --nproc_per_node=1 MHC_train.py --mode independent
python MHC_test.py --mode independent
```

### 2.3 pHLA-II Binding

Precompute peptide embeddings before training or evaluation:

```bash
cd Downstream/pMHC-II
python precompute_peptide_embeddings.py --split all --max-length 34
```

Train all cross-validation folds and evaluate both warm- and cold-start test sets:

```bash
python train.py --mode cv_train --folds all
python test.py --eval-set both --folds all
```

To evaluate only one split, use either:

```bash
python test.py --eval-set warm --folds all
python test.py --eval-set cold --folds all
```

### 2.4 pTCR Recognition

#### Seen and Unseen Evaluation

Train on the seen-data folds:

```bash
cd Downstream/pTCR2
torchrun --standalone --nproc_per_node=1 TCR_train.py \
  --cv_dir ./data/Seen_5fold_splits \
  --save_dir ../trained_model/pTCR3/Seen \
  --embed_backend AntigenLM
```

Evaluate the seen folds and unseen set:

```bash
python TCR_test.py \
  --cv_dir ./data/Seen_5fold_splits \
  --weights_dir ../trained_model/pTCR3/Seen \
  --out_dir ../result/pTCR3/AntigenLM_Seen \
  --embed_backend AntigenLM

python TCR_test.py \
  --weights_dir ../trained_model/pTCR3/Seen \
  --out_dir ../result/pTCR3/AntigenLM_Unseen \
  --unseen_csv ./data/Unseen.csv \
  --embed_backend AntigenLM
```

#### CMA and COVID-19 Evaluation

Train on the CMA folds:

```bash
cd Downstream/pTCR2
torchrun --standalone --nproc_per_node=1 TCR_train.py \
  --cv_dir ./data/CMA_5fold_splits \
  --save_dir ../trained_model/pTCR3/CMA \
  --embed_backend AntigenLM
```

Evaluate the CMA folds and COVID-19 set:

```bash
python TCR_test.py \
  --cv_dir ./data/CMA_5fold_splits \
  --weights_dir ../trained_model/pTCR3/CMA \
  --out_dir ../result/pTCR3/AntigenLM_CMA \
  --embed_backend AntigenLM

python TCR_test.py \
  --weights_dir ../trained_model/pTCR3/CMA \
  --out_dir ../result/pTCR3/AntigenLM_CMA_Covid \
  --independent_csv ./data/Covid_set.csv \
  --embed_backend AntigenLM
```

### 2.5 B-cell Epitope Prediction

```bash
cd Downstream/B_cell_epitope
python train.py
python test.py
```

## 3) Demo

The demo performs protective-antigen prediction for 100 protein sequences with the released AntigenLM encoder and the cluster-aware fold 1 classifier.

### 3.1 Required Files

- Input: `demo/input/demo_input.csv`
- AntigenLM encoder: `LLM/AntigenLM/`
- Classifier checkpoint: `Downstream/trained_model/protective_antigen/30_similarity/fold_1_seed22_AntigenLM.pt`

The input CSV contains `ID`, `sequence`, and `label` columns. Labels are used only to evaluate the example predictions; `1` denotes a protective antigen and `0` a non-protective antigen.

### 3.2 Run the Demo

From the repository root:

```bash
conda activate AntigenLM
bash demo/run_demo.sh
```

### 3.3 Run on Your Own Data

Prepare a CSV file containing the `ID`, `sequence`, and `label` columns, where `label` is `1` for a protective antigen and `0` otherwise. Then run the following command from the repository root, replacing `/path/to/your_data.csv` with your input file:

```bash
conda activate AntigenLM
python demo/predict.py \
  --input /path/to/your_data.csv \
  --model-dir LLM/AntigenLM \
  --classifier Downstream/trained_model/protective_antigen/30_similarity/fold_1_seed22_AntigenLM.pt \
  --output-dir demo/output/your_data
```

Predictions and evaluation metrics will be written to `demo/output/your_data/`.

### 3.4 Output

Predictions are written to:

```text
demo/output/AntigenLM_cluster_aware_fold_1_pred_results.csv
```

Metrics are written to `demo/output/AntigenLM_cluster_aware_fold_1_metrics.csv`. The prediction columns are `id`, `y_true`, `y_pred`, and `y_score`. Reference predictions and metrics are provided in `demo/reference/`. Minor score differences can occur across hardware and CUDA versions.

Runtime depends on sequence lengths and the available GPU. See `demo/README.md` for further details.

### 3.5 Expected Runtime

On the tested Ubuntu 20.04.6 LTS system with an NVIDIA A100-SXM4-80GB GPU, the demo processes 100 protein sequences in approximately 1 minute. This runtime includes model loading, embedding extraction, and classifier inference, but excludes model and checkpoint download time.

CPU-only execution has not been benchmarked and is not recommended for this demo.

## 4) Reproduction of Main Results

The repository includes released predictions and summary metrics under `Result/`. Use the following workflow to reproduce the principal AntigenLM results.

### 4.1 Prepare Models and Data

1. Create the environment described in [Installation](#installation).
2. Place the AntigenLM encoder in `LLM/AntigenLM/`.
3. Place released task checkpoints in `Downstream/trained_model/`, preserving the downloaded directory structure.
4. Confirm that each task dataset is present in the corresponding `Downstream/<task>/data/` directory.

Using the released downstream checkpoints reproduces evaluation without retraining. To reproduce the full experimental pipeline, run the training command before the test command for every task in [Downstream](#2-downstream).


### 4.2 Figure-ready Results

Data files used for the main figures are collected under `Downstream/Result/`:

| Directory | Contents |
| --- | --- |
| `Downstream/Result/Fig 2/` | Protective-antigen predictions and summary metrics, including the micro-dataset and independent bacterial and viral datasets |
| `Downstream/Result/Fig 3/` | pHLA-I predictions and summary metrics for the micro-dataset and independent MUNIS dataset |
| `Downstream/Result/Fig 4/` | pTCR predictions and summary metrics for All, COVID-19, Seen, Unseen, and independent evaluations |
| `Downstream/Result/Fig 6/` | Ablation-study fold metrics and the combined ablation summary |

The multi-fold prediction workbooks contain one worksheet per fold. The accompanying CSV files provide method-level summaries and independent-test results.


## License

Unless otherwise stated, the source code in this repository is licensed under
the MIT License. See the [LICENSE](LICENSE) file for details.

The pretrained model checkpoints and datasets are distributed separately and
are subject to the license terms specified in their respective repositories.

![AntigenLM overview](Fig%201.png)
