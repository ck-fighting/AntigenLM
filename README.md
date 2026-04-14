# AntigenLM Codebase Overview

## Installation

1. Clone the repository.
   ```bash
   git clone https://github.com/ck-fighting/AntigenLM.git
   cd AntigenLM
   ```

2. Create a virtual environment by conda.
   ```bash
   conda env create -f environment.yml
   conda activate AntigenLM
   ```

---

## Overview

This repository contains two main parts:

- `AntigenLM_training/`: pretraining and fine-tuning pipelines for microLM → PathogLM → antigenLM.
- `Downstream/`: downstream tasks (protective antigen prediction, pTCR recognition, pMHC recognition) using pretrained embeddings.

> [!IMPORTANT]
> - **Training**: Before running any training scripts, you must download the datasets from the links provided below and place them in the `AntigenLM_training/dataset/` directory.
> - **Downstream Tasks**: To run downstream tasks, you must first download the **AntigenLM** model weights from https://huggingface.co/cckai2017/AntigenLM and place them in the `LLM/` directory.

## 1) AntigenLM_training


### 1.1 Structure

- `bert_pretrain_main.py`: MLM pretraining (RoFormer) with HuggingFace Trainer + DeepSpeed.
- `bert_finetuning_Freeze.py`: freeze most layers and fine-tune last layers to get PathogLM or antigenLM.
- `bert_finetuning_LoRa.py`: (Optional) script for LoRA experiments.
- `data/`: dataset loaders and collators.
- `bert_data_prepare/`: tokenizer utils, vocab, blosum matrix.
- `config/`: JSON configs used by the scripts.

### 1.2 Datasets (Hugging Face)

- [dataset_micro.fasta](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/dataset_micro.fasta): pretraining FASTA for microLM (raw protein sequences).
- [antigen_seq_ss_2.csv](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/antigen_seq_ss_2.csv): antigen sequences with secondary-structure labels (sequence, second_structure) for antigenLM fine-tuning.
- [pathogen_seq_ss.csv](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/pathogen_seq_ss.csv): pathogen sequences with secondary-structure labels (sequence, second_structure) for PathogLM freeze fine-tuning.


### 1.3 Trained_model(Hugging Face)

- [AntigenLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/Result_antigenLM_300M_SS): final model used directly for downstream embeddings.
- [PathogLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/Result_PathogLM_300M_SS): intermediate model obtained by freeze fine-tuning from microLM.
- [MicroLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/Result_microLM_300M): base pretrained model.

### 1.4 Configs

- `AntigenLM_training/config/bert_pretrain_microLM.json`: microLM pretraining config (FASTA input, RoFormer size, DeepSpeed and output checkpoint location).
- `AntigenLM_training/config/bert_finetrain_microLM_freeze.json`: freeze fine-tuning config to train PathogLM from microLM (CSV with sequence + second_structure).
- `AntigenLM_training/config/bert_finetrain_pathogLM_freeze.json`: freeze fine-tuning config to train AntigenLM from PathogLM (CSV with sequence + second_structure).

These configs define pretrained_model_path, dataset paths, and output save_dir.

### 1.5 Run commands

Pretrain microLM (DeepSpeed):
```bash
cd AntigenLM_training
deepspeed --num_gpus=2 bert_pretrain_main.py -c config/bert_pretrain_microLM.json
```

Freeze fine-tune to PathogLM:
```bash
cd AntigenLM_training
python bert_finetuning_Freeze.py -c config/bert_finetrain_microLM_freeze.json
```

Freeze fine-tune PathogLM → antigenLM:
```bash
cd AntigenLM_training
python bert_finetuning_Freeze.py -c config/bert_finetrain_pathogLM_freeze.json
```

Notes:
- `bert_pretrain_main.py` hard-codes CUDA_VISIBLE_DEVICES="0,1" at the top. Adjust in the script or use --device in parse_config.py only if you remove that hard-coded line.

## 2) Downstream

> [!NOTE]
> The trained checkpoints for the three downstream tasks (protective antigen prediction, pMHC recognition, and pTCR recognition) have already been prepared and are available on Hugging Face:
> 
> https://huggingface.co/cckai2017/Downstream_trained_model/tree/main
> 
> Please download the required files and place them under `Downstream/trained_model/` using the directory structure expected by the scripts.


### 2.1 Protective antigen classification

Train (example):
```bash
cd Downstream/protective_antigen
python protective_antigen_train.py  --data_dir ./data --save_dir ../trained_model/protective_antigen --embed_backend AntigenLM
```

Test (example):
```bash
cd Downstream/protective_antigen
python protective_antigen_test.py --data_dir ./data --weights_dir ../trained_model/protective_antigen --out_dir ../result/protective_antigen --embed_backend AntigenLM
```



### 2.2 pMHC (HLA) binding

Train (example, 2 GPUs):
```bash
cd Downstream/pMHC
torchrun --nproc_per_node=2 MHC_train.py --cv_dir ./data/cv_splits --save_dir ../trained_model/pMHC --embed_backend AntigenLM
```

Test (example):
```bash
cd Downstream/pMHC
python MHC_test.py --data_csv ./data/micro_test_set.csv --weights_dir ../trained_model/pMHC --out_dir ../result/pMHC --embed_backend AntigenLM
```


### 2.3 pTCR binding

#### 2.3.1 Majority

Train:
```bash
cd Downstream/pTCR
torchrun --nproc_per_node=2 TCR_train.py --cv_dir ./data/cv_Majority/ --save_dir ../trained_model/pTCR/Majority/test --embed_backend AntigenLM
```

Test (Majority test set):
```bash
cd Downstream/pTCR
python TCR_test.py --mode eval --data_csv ./data/majority_test_set.csv --weights_dir ../trained_model/pTCR/Majority --out_dir ../result/pTCR/Majority --embed_backend AntigenLM
```

Test (Covid test set):
```bash
cd Downstream/pTCR
python TCR_test.py --mode eval --data_csv ./data/covid_set.csv --weights_dir ../trained_model/pTCR/Majority --out_dir ../result/pTCR/Covid --embed_backend AntigenLM
```

#### 2.3.2 Zero-shot & Few-shot

Train:
```bash
cd Downstream/pTCR
torchrun --nproc_per_node=2 TCR_train.py --cv_dir ./data/cv_backbone --save_dir ../trained_model/pTCR/backbone --embed_backend AntigenLM
```

Test (Zero-shot):
```bash
cd Downstream/pTCR
python TCR_test.py --mode eval --data_csv ./data/zero_shot_set.csv --weights_dir ../trained_model/pTCR/backbone --out_dir ../result/pTCR/Zero_shot --embed_backend AntigenLM
```

Test (Few-shot):
```bash
cd Downstream/pTCR
 python TCR_test.py --mode fewshot --support_csv ./data/few_shot_support_set.csv --query_csv ./data/few_shot_query_set.csv --finetune --weights_dir ../trained_model/pTCR/backbone --out_dir ../result/pTCR/Few_shot --embed_backend AntigenLM
```

Make sure paths in configs point to existing data and model checkpoints on your machine.

![Fig 1](Fig%201.png)