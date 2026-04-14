# AntigenLM Training Datasets

This directory is intended to store the datasets for pretraining and fine-tuning AntigenLM. Due to file size limitations, the datasets are hosted on Hugging Face.

## Download Links

- [**dataset_micro.fasta**](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/dataset_micro.fasta)
  - **Description**: Pretraining FASTA for microLM (raw protein sequences).
- [**antigen_seq_ss_2.csv**](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/antigen_seq_ss_2.csv)
  - **Description**: Antigen sequences with secondary-structure labels (`sequence`, `second_structure`) for antigenLM fine-tuning.
- [**pathogen_seq_ss.csv**](https://huggingface.co/datasets/cckai2017/AntigenLM/blob/main/pathogen_seq_ss.csv)
  - **Description**: Pathogen sequences with secondary-structure labels (`sequence`, `second_structure`) for PathogLM freeze fine-tuning.

## Usage

After downloading, place the files in this directory (`AntigenLM_training/dataset/`) to ensure the training configurations can locate them.
