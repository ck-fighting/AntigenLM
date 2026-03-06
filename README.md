# AntigenLM Codebase Overview



This repository contains two main parts:

- `AntigenLM_training/`: pretraining and fine-tuning pipelines for microLM → PathogLM → antigenLM.
- `Downstream/`: downstream tasks (protective antigen classification, pTCR, pMHC) using pretrained embeddings.



## 1) AntigenLM_training

### Structure

- `bert_pretrain_main.py`: MLM pretraining (RoFormer) with HuggingFace Trainer + DeepSpeed.
- `bert_finetuning_LoRa.py`: LoRA fine-tuning to get PathogLM.
- `bert_finetuning_Freeze.py`: freeze most layers and fine-tune last layers to get antigenLM.
- `data/`: dataset loaders and collators.
- `bert_data_prepare/`: tokenizer utils, vocab, blosum matrix.
- `config/`: JSON configs used by the scripts.

### Datasets (Hugging Face)

- [`dataset_micro.fasta`](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/dataset_micro.fasta): pretraining FASTA for microLM (raw protein sequences).
- [`antigen_seq_ss_2.csv`](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/antigen_seq_ss_2.csv): antigen sequences with secondary-structure labels (`sequence`, `second_structure`) for antigenLM fine-tuning.
- [`pathogen_seq_ss.csv`](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/pathogen_seq_ss.csv): pathogen sequences with secondary-structure labels (`sequence`, `second_structure`) for PathogLM LoRA fine-tuning.


### Trained_model

- [AntigenLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/Result_antigenLM_300M_SS): final model used directly for downstream embeddings.
- [PathogLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/Result_PathogLM_300M_SS): intermediate model obtained by LoRA fine-tuning from microLM.
- [MicroLM](https://huggingface.co/cckai2017/AntigenLM/tree/main/Result_microLM_300M): base pretrained model.

### Configs

- `AntigenLM_training/config/bert_pretrain_microLM.json`: microLM pretraining config (FASTA input, RoFormer size, DeepSpeed and output checkpoint location).
- `AntigenLM_training/config/bert_finetrain_microLM_Lora.json`: LoRA fine-tuning config to train PathogLM from microLM (CSV with `sequence` + `second_structure`).
- `AntigenLM_training/config/bert_finetrain_pathogLM_freeze.json`: freeze fine-tuning config to train antigenLM from PathogLM (CSV with `sequence` + `second_structure`).

These configs define `pretrained_model_path`, dataset paths, and output `save_dir`.

### Run commands

Pretrain microLM (DeepSpeed):
```bash
deepspeed --num_gpus=2 AntigenLM_training/bert_pretrain_main.py -c AntigenLM_training/config/bert_pretrain_microLM.json
```

LoRA fine-tune to PathogLM:
```bash
python AntigenLM_training/bert_finetuning_LoRa.py -c AntigenLM_training/config/bert_finetrain_microLM_Lora.json
```

Freeze fine-tune PathogLM → antigenLM:
```bash
python AntigenLM_training/bert_finetuning_Freeze.py -c AntigenLM_training/config/bert_finetrain_pathogLM_freeze.json
```

Notes:
- `bert_pretrain_main.py` hard-codes `CUDA_VISIBLE_DEVICES="0,1"` at the top. Adjust in the script or use `--device` in `parse_config.py` only if you remove that hard-coded line.

## 2) Downstream

### 2.1 Protective antigen classification

Entry points:
- Train: `Downstream/protective_antigen/train.py`
- Test: `Downstream/protective_antigen/test.py`

Key settings:
- `model_type` : set to `AntigenLM`.
- `data_dir`: `Downstream/protective_antigen/data/.`
- `trained_model`: `trained_model/protective_antigen/.`
- `All Results`: saved in `Result/protective_antigen/`

Train:
```bash
python Downstream/protective_antigen/train.py
```

Test:
```bash
python Downstream/protective_antigen/test.py
```



### 2.2 pTCR binding

Entry points:
- Train: `Downstream/pTCR/TCR_train.py` (DDP required)
- Test: `Downstream/pTCR/TCR_test.py`

Default data layout:
- `Downstream/pTCR/data/cv_LLM/train_fold_*.csv`
- `Downstream/pTCR/data/cv_LLM/val_fold_*.csv`

Train (example, 2 GPUs):
```bash
torchrun --standalone --nproc_per_node=2 Downstream/pTCR/TCR_train.py \
  --cv_dir Downstream/pTCR/data/cv_LLM \
  --embed_backend antigenLM_withoutSlidingwindow
```

Test (example, eval mode):
```bash
python Downstream/pTCR/TCR_test.py \
  --mode eval \
  --data_csv Downstream/pTCR/data/covid_set.csv \
  --weights_dir Downstream/trained_model/pTCR/Majority \
  --embed_backend antigenLM
```

### 2.3 pMHC (HLA) binding

Entry points:
- Train: `Downstream/pMHC/MHC_train.py` (DDP required)
- Test: `Downstream/pMHC/MHC_test.py`

Default data layout:
- `Downstream/pMHC/data/cv_splits/train_fold_*.csv`
- `Downstream/pMHC/data/cv_splits/val_fold_*.csv`

Train (example, 2 GPUs):
```bash
torchrun --standalone --nproc_per_node=2 Downstream/pMHC/MHC_train.py \
  --cv_dir Downstream/pMHC/data/cv_splits \
  --embed_backend microLM
```

Test (example):
```bash
python Downstream/pMHC/MHC_test.py \
  --data_csv Downstream/pMHC/data/micro_set.csv \
  --weights_dir Downstream/trained_model/pMHC \
  --embed_backend antigenLM
```

## Dependencies 

- `torch==2.1.0+cu121`, `torchvision==0.16.0+cu121`, `torchaudio==2.1.0+cu121`
- `transformers==4.46.3`, `tokenizers==0.20.3`, `sentencepiece==0.2.0`, `safetensors==0.5.3`, `huggingface-hub==0.36.2`, `accelerate==1.0.1`
- `deepspeed==0.10.3`
- `peft==0.13.2`
- `fair-esm==2.0.0`
- `numpy==1.24.1`, `pandas==2.0.3`, `scipy==1.10.1`, `scikit-learn==1.3.2`, `tqdm==4.67.1`
- `biopython==1.83`

Make sure paths in configs point to existing data and model checkpoints on your machine.

![Fig 1](Fig%201.png)