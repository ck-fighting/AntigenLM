#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import collections
import os
import random
import shutil
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput

import data.bert_pretrain_maa_dataset as module_data
from data.utility import DatasetSplit
from esmc_pretrain_main import EsmcMlmDataCollator
from model.metric import MAA_metrics
from parse_config import ConfigParser


os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/hf-modules")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch-extensions")
os.environ.setdefault("MASTER_PORT", "29508")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_except_last_blocks(model, last_n_layers, logger=None):
    for param in model.parameters():
        param.requires_grad = False

    blocks = model.backbone.transformer.blocks
    first_trainable = max(0, len(blocks) - int(last_n_layers))
    for block in blocks[first_trainable:]:
        for param in block.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    if logger:
        logger.info(
            f"Training full backbone blocks {first_trainable}-{len(blocks) - 1}; "
            f"trainable tensors: {trainable_tensors}; "
            f"trainable params: {trainable_params:,} / {total_params:,}"
        )
    return model, first_trainable


def enable_frozen_prefix_no_grad(model, first_trainable_layer, logger=None):
    def forward_with_frozen_prefix_no_grad(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        sequence_tokens=None,
        sequence_id=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        if sequence_tokens is not None:
            input_ids = sequence_tokens
        if sequence_id is not None:
            attention_mask = sequence_id
        if input_ids is None:
            raise ValueError("input_ids or sequence_tokens must be provided")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)

        if getattr(self.backbone, "_use_flash_attn", False):
            raise RuntimeError("This Hugging Face wrapper expects use_flash_attn=False")

        with torch.no_grad():
            x = self.backbone.embed(input_ids)
            *batch_dims, _ = x.shape
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
            for block in self.backbone.transformer.blocks[:first_trainable_layer]:
                x = block(x, attention_mask, None, None, chain_id)

        for block in self.backbone.transformer.blocks[first_trainable_layer:]:
            x = block(x, attention_mask, None, None, chain_id)

        x = self.backbone.transformer.norm(x)
        hidden_states = (x,) if output_hidden_states else None
        logits = self.backbone.sequence_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (hidden_states,)
            return ((loss,) + output) if loss is not None else output
        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=hidden_states)

    if first_trainable_layer > 0:
        model.forward = types.MethodType(forward_with_frozen_prefix_no_grad, model)
        if logger:
            logger.info(
                f"Enabled no_grad forward for frozen prefix blocks 0-{first_trainable_layer - 1}; "
                f"autograd starts at block {first_trainable_layer}."
            )
    return model


def build_model(config, logger):
    model = AutoModelForMaskedLM.from_pretrained(
        config["pretrained_model_path"],
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    last_n_layers = config["trainer"].get("last_n_layers", 8)
    model, first_trainable = freeze_except_last_blocks(model, last_n_layers, logger=logger)
    return enable_frozen_prefix_no_grad(model, first_trainable, logger=logger)


def build_metric_token_list(dataset, model):
    tokens = dataset.get_token_list()
    model_vocab_size = int(getattr(model.config, "vocab_size", len(tokens)))
    if len(tokens) < model_vocab_size:
        tokens = list(tokens) + [f"<unused_{idx}>" for idx in range(len(tokens), model_vocab_size)]
    return tokens


def export_runtime_files(pretrained_model_path, output_dir):
    source_dir = Path(pretrained_model_path)
    output_dir = Path(output_dir)
    for filename in (
        "configuration_antigenlm.py",
        "modeling_antigenlm.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "model_card.json",
        "full_bias_conversion.json",
    ):
        source = source_dir / filename
        if source.exists():
            shutil.copyfile(source, output_dir / filename)


def main(config):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
        import deepspeed

        deepspeed.init_distributed("nccl")

    logger = config.get_logger("train")
    seed = config["dataset"]["args"]["seed"]
    set_seed(seed)

    config["dataset"]["args"]["config"] = config
    config["dataset"]["args"]["logger"] = logger
    config["dataset"]["args"]["tokenizer_name"] = "esmc"
    config["dataset"]["args"]["token_length_list"] = ""
    config["dataset"]["args"]["vocab_dir"] = ""

    dataset = config.init_obj("dataset", module_data)
    full_train_dataset = dataset.get_dataset()

    holdout = config["dataset"]["args"].get("test_split")
    test_dataset = None
    if holdout is not None:
        assert 0.0 < holdout < 1.0, "holdout must be between 0 and 1"
        test_dataset = DatasetSplit(
            logger=logger,
            full_dataset=full_train_dataset,
            split="test",
            valid=0,
            test=holdout,
        )
        train_dataset = DatasetSplit(
            logger=logger,
            full_dataset=full_train_dataset,
            split="train",
            valid=0,
            test=holdout,
        )
    else:
        train_dataset = full_train_dataset

    tokenizer = dataset.get_tokenizer()
    data_collator = EsmcMlmDataCollator(
        tokenizer=tokenizer,
        mlm_probability=config["trainer"].get("mlm_probability", 0.15),
    )

    config.config["deepspeed"]["train_micro_batch_size_per_gpu"] = config["trainer"]["batch_size"]
    training_args = TrainingArguments(
        output_dir=config["trainer"]["save_dir"],
        overwrite_output_dir=True,
        num_train_epochs=config["trainer"]["epochs"],
        max_steps=config["trainer"].get("max_steps", -1),
        per_device_train_batch_size=config["trainer"]["batch_size"],
        per_device_eval_batch_size=int(config["trainer"]["batch_size"]),
        learning_rate=config["trainer"]["lr"],
        warmup_ratio=config["trainer"]["warmup"],
        weight_decay=config["trainer"]["weight_decay"],
        evaluation_strategy="steps" if test_dataset is not None else "no",
        eval_steps=config["trainer"].get("eval_steps", 5000),
        save_steps=config["trainer"].get("save_steps", 5000),
        save_total_limit=config["trainer"].get("save_total_limit", 5),
        eval_accumulation_steps=config["trainer"].get("eval_accumulation_steps"),
        logging_strategy="steps",
        logging_steps=config["trainer"]["logging_steps"],
        save_strategy="steps",
        dataloader_num_workers=config["trainer"].get("dataloader_num_workers", 8),
        dataloader_pin_memory=config["trainer"].get("dataloader_pin_memory", True),
        dataloader_persistent_workers=(
            config["trainer"].get("dataloader_persistent_workers", True)
            and config["trainer"].get("dataloader_num_workers", 8) > 0
        ),
        dataloader_prefetch_factor=config["trainer"].get("dataloader_prefetch_factor", 2),
        load_best_model_at_end=test_dataset is not None,
        metric_for_best_model=config["trainer"].get("metric_for_best_model", "eval_acc"),
        greater_is_better=True,
        logging_dir=config._log_dir,
        fp16=config["deepspeed"].get("fp16", {}).get("enabled", False),
        deepspeed=config["deepspeed"],
        gradient_accumulation_steps=config["deepspeed"].get("gradient_accumulation_steps", 1),
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
        no_cuda=False,
        skip_memory_metrics=True,
    )

    pretrained_model_path = config["pretrained_model_path"]
    model = build_model(config, logger)

    maa_metrics = MAA_metrics(
        token_with_special_list=build_metric_token_list(dataset, model),
        blosum_dir=config["metrics"]["blosum_dir"],
        blosum=config["metrics"]["blosum"],
    )

    callbacks = []
    if test_dataset is not None and config["trainer"].get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["trainer"]["early_stopping_patience"]
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=maa_metrics.compute_metrics if test_dataset is not None else None,
        callbacks=callbacks,
    )

    if config.resume is not None:
        trainer.train(resume_from_checkpoint=config.resume)
    else:
        trainer.train()

    trainer.save_model(config._save_dir)
    tokenizer.save_pretrained(config._save_dir)
    export_runtime_files(pretrained_model_path, config._save_dir)
    logger.info(f"Last-8-layer full fine-tuned checkpoint saved at {config._save_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(
        description="Fine-tune only the last N ESM-C-style backbone blocks on FASTA MLM data."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/bert_finetune_MicroLM_esmc_style_full_bias_last8_layers.json",
        type=str,
    )
    parser.add_argument("-r", "--resume", default=None, type=str)
    parser.add_argument("-d", "--device", default=None, type=str)
    parser.add_argument("-local_rank", "--local_rank", default=0, type=int)

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="trainer;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="trainer;batch_size"),
        CustomArgs(["--steps", "--max_steps"], type=int, target="trainer;max_steps"),
        CustomArgs(["--last_n_layers", "--last-n-layers"], type=int, target="trainer;last_n_layers"),
    ]

    parsed_config = ConfigParser.from_args(parser, options)
    main(parsed_config)
