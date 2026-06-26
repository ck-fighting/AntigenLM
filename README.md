# AntigenLM Codebase Overview

## Installation

1. Clone the repository.
   ```bash
   git clone https://github.com/ck-fighting/AntigenLM.git
   cd AntigenLM
   ```

2. Create a virtual environment by conda.
   ```bash
   conda create -n AntigenLM python=3.8 -y
   conda activate AntigenLM
   ```

3. Install dependencies.
   ```bash
   pip install torch torchvision torchaudio
   pip install transformers deepspeed accelerate safetensors
   pip install pandas numpy scipy scikit-learn tqdm openpyxl biopython
   ```

---

## Overview

This repository contains two main parts:

- `Pre-training/`: pretraining and fine-tuning pipelines for MicroLM -> PathogLM -> AntigenLM.
- `Downstream/`: downstream tasks (protective antigen prediction, pTCR recognition, pHLA-I recognition, pHLA-II recognition, and B-cell epitope prediction) using pretrained embeddings.

> [!IMPORTANT]
> - **Training**: Before running any training scripts, you must download the datasets from the links provided below and place them in the `Pre-training/dataset/` directory.
> - **Downstream Tasks**: To run downstream tasks, you must first download the **AntigenLM** model weights from https://huggingface.co/cckai2017/AntigenLM and place them in the `LLM/AntigenLM/` directory.

## 1) Pre-training

### 1.1 Structure

- `esmc_pretrain_main.py`: MicroLM pretraining with DeepSpeed.
- `bert_finetuning_esmc_style_last8_layers.py`: fine-tuning script used for MicroLM and PathogLM stages.
- `data/`: dataset loaders and collators.
- `bert_data_prepare/`: tokenizer and data preparation utilities.
- `config/`: JSON configs used by the scripts.
- `dataset/`: downloaded pretraining and fine-tuning datasets.

### 1.2 Datasets (Hugging Face)

- [antigen_seq_ss.csv](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/antigen_seq_ss.csv): antigen sequences with secondary-structure labels (`sequence`, `second_structure`) for AntigenLM fine-tuning.
- [pathogen_seq.fasta](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/pathogen_seq.fasta): pathogen FASTA for PathogLM fine-tuning.
- [dataset_micro.fasta.fasta](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/dataset_micro.fasta.fasta): pretraining FASTA for MicroLM (raw protein sequences).

### 1.3 Trained Models (Hugging Face)

- [AntigenLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/AntigenLM): final model used directly for downstream embeddings.
- [PathogLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/PathogLM): intermediate model obtained by fine-tuning from MicroLM.
- [MicroLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/MicroLM): base pretrained model.

### 1.4 Configs

- `Pre-training/config/bert_pretrain_esmc_300m.json`: MicroLM pretraining config.
- `Pre-training/config/bert_finetune_MicroLM_esmc_style_full_bias_last8_layers.json`: fine-tuning config to train PathogLM from MicroLM.
- `Pre-training/config/bert_finetune_PathogLM_esmc_style_full_bias_last8_layers.json`: fine-tuning config to train AntigenLM from PathogLM.

These configs define pretrained model paths, dataset paths, and output checkpoint locations.

### 1.5 Run Commands

Pretrain MicroLM:
```bash
cd Pre-training
deepspeed --num_gpus=4 esmc_pretrain_main.py -c config/bert_pretrain_esmc_300m.json
```

Fine-tune MicroLM to PathogLM:
```bash
cd Pre-training
deepspeed --num_gpus=4 bert_finetuning_esmc_style_last8_layers.py -c config/bert_finetune_MicroLM_esmc_style_full_bias_last8_layers.json
```

Fine-tune PathogLM to AntigenLM:
```bash
cd Pre-training
deepspeed --num_gpus=4 bert_finetuning_esmc_style_last8_layers.py -c config/bert_finetune_PathogLM_esmc_style_full_bias_last8_layers.json
```

## 2) Downstream

> [!NOTE]
> The trained checkpoints for downstream tasks are available on Hugging Face:
>
> https://huggingface.co/cckai2017/Downstream_trained_model/tree/main
>
> Please download the required files and place them under `Downstream/trained_model/` using the directory structure expected by the scripts.

### 2.1 Protective Antigen Classification

Train:
```bash
cd Downstream/protective_antigen
python protective_antigen_train.py
```

Test:
```bash
cd Downstream/protective_antigen
python protective_antigen_test.py
```

### 2.2 pHLA-I Binding

Train:
```bash
cd Downstream/pMHC-I
torchrun --standalone --nproc_per_node=1 MHC_train.py
```

Test:
```bash
cd Downstream/pMHC-I
python MHC_test.py
```

### 2.3 pHLA-II Binding

Precompute peptide embeddings:
```bash
cd Downstream/pMHC-II
python precompute_peptide_embeddings.py
```

Train:
```bash
cd Downstream/pMHC-II
python train.py
```

Test:
```bash
cd Downstream/pMHC-II
python test.py
```

### 2.4 pTCR Binding

#### 2.4.1 Seen and Unseen

Train:
```bash
cd Downstream/pTCR2
torchrun --standalone --nproc_per_node=1 TCR_train.py --cv_dir ./data/Seen_5fold_splits --save_dir ../trained_model/pTCR3/Seen --embed_backend AntigenLM
```

Test (Seen 5-fold test set):
```bash
cd Downstream/pTCR2
python TCR_test.py --cv_dir ./data/Seen_5fold_splits --weights_dir ../trained_model/pTCR3/Seen --out_dir ../result/pTCR3/AntigenLM_Seen --embed_backend AntigenLM
```

Test (Unseen test set):
```bash
cd Downstream/pTCR2
python TCR_test.py --weights_dir ../trained_model/pTCR3/Seen --out_dir ../result/pTCR3/AntigenLM_Unseen --unseen_csv ./data/Unseen.csv --embed_backend AntigenLM
```

#### 2.4.2 CMA

Train:
```bash
cd Downstream/pTCR2
torchrun --standalone --nproc_per_node=1 TCR_train.py --cv_dir ./data/CMA_5fold_splits --save_dir ../trained_model/pTCR3/CMA --embed_backend AntigenLM
```

Test (CMA 5-fold test set):
```bash
cd Downstream/pTCR2
python TCR_test.py --cv_dir ./data/CMA_5fold_splits --weights_dir ../trained_model/pTCR3/CMA --out_dir ../result/pTCR3/AntigenLM_CMA --embed_backend AntigenLM
```

Test (Covid test set under CMA):
```bash
cd Downstream/pTCR2
python TCR_test.py --weights_dir ../trained_model/pTCR3/CMA --out_dir ../result/pTCR3/AntigenLM_CMA_Covid --independent_csv ./data/Covid_set.csv --embed_backend AntigenLM
```

### 2.5 B-cell Epitope Prediction

Train:
```bash
cd Downstream/B_cell_epitope
python train.py
```

Test:
```bash
cd Downstream/B_cell_epitope
python test.py
```

Make sure paths in configs and scripts point to existing data and model checkpoints on your machine.

![Fig 1](Fig%201.png)
