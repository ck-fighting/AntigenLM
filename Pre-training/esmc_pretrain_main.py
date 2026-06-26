#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import collections
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch-extensions")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import EarlyStoppingCallback, PretrainedConfig, PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import MaskedLMOutput

import data.bert_pretrain_maa_dataset as module_data
from data.utility import DatasetSplit
from model.metric import MAA_metrics
from parse_config import ConfigParser


class AntigenLMConfig(PretrainedConfig):
    model_type = "antigenlm"

    def __init__(
        self,
        d_model=960,
        n_heads=15,
        n_layers=30,
        vocab_size=64,
        use_flash_attn=False,
        disable_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.use_flash_attn = use_flash_attn
        self.disable_bias = disable_bias


def remove_linear_and_layernorm_biases(module):
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.LayerNorm)) and getattr(child, "bias", None) is not None:
            child.bias = None


def patch_esmc_tokenizer_for_transformers_compat():
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    def make_special_property(name):
        storage_name = "_" + name

        def getter(self):
            return getattr(self, storage_name, None)

        def setter(self, value):
            setattr(self, storage_name, value)

        return property(getter, setter)

    for token_name in ("cls_token", "eos_token", "mask_token", "pad_token", "bos_token"):
        prop = getattr(EsmSequenceTokenizer, token_name, None)
        if isinstance(prop, property) and prop.fset is None:
            setattr(EsmSequenceTokenizer, token_name, make_special_property(token_name))


class AntigenLMForMaskedLM(PreTrainedModel):
    config_class = AntigenLMConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = False

    def __init__(self, config):
        super().__init__(config)
        patch_esmc_tokenizer_for_transformers_compat()
        from esm.models.esmc import ESMC
        from esm.pretrained import get_esmc_model_tokenizers

        self.backbone = ESMC(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=config.use_flash_attn,
        )
        if getattr(config, "disable_bias", False):
            remove_linear_and_layernorm_biases(self.backbone)

    @property
    def tokenizer(self):
        return self.backbone.tokenizer

    def forward(
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

        x = self.backbone.embed(input_ids)
        *batch_dims, _ = x.shape
        chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
        for block in self.backbone.transformer.blocks:
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


class EsmcMlmDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.special_ids = set(tokenizer.all_special_ids)

    def __call__(self, examples):
        input_ids = [example["input_ids"] for example in examples]
        max_len = max(ids.size(0) for ids in input_ids)

        batch_input_ids = []
        attention_mask = []
        for ids in input_ids:
            pad_len = max_len - ids.size(0)
            if pad_len:
                ids = torch.cat(
                    [
                        ids,
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            batch_input_ids.append(ids)
            attention_mask.append(ids.ne(self.pad_token_id))

        input_ids = torch.stack(batch_input_ids)
        attention_mask = torch.stack(attention_mask)
        labels = input_ids.clone()

        probability = torch.full(labels.shape, self.mlm_probability)
        special_mask = torch.zeros(labels.shape, dtype=torch.bool)
        for token_id in self.special_ids:
            special_mask |= labels.eq(token_id)
        probability.masked_fill_(special_mask, value=0.0)

        masked_indices = torch.bernoulli(probability).bool()
        if not masked_indices.any():
            candidates = (~special_mask).nonzero(as_tuple=False)
            if candidates.numel() > 0:
                picked = candidates[torch.randint(candidates.size(0), (1,)).item()]
                masked_indices[picked[0], picked[1]] = True

        labels[~masked_indices] = -100
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_antigenlm_model(logger, model_args, vocab_size, pad_token_id):
    logger.info("Building AntigenLM with an ESM-C-style backbone from random initialization")
    config = AntigenLMConfig(
        **model_args,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        bos_token_id=0,
        cls_token_id=0,
        eos_token_id=2,
        mask_token_id=32,
        unk_token_id=3,
    )
    config.architectures = ["AntigenLMForMaskedLM"]
    config.auto_map = {
        "AutoConfig": "configuration_antigenlm.AntigenLMConfig",
        "AutoModelForMaskedLM": "modeling_antigenlm.AntigenLMForMaskedLM",
    }
    config.architecture_family = "ESM-C-style protein language model"
    config.parameter_initialization = "random"
    config.pretraining_source = "from_scratch"
    config.base_pretrained_checkpoint = None
    config.pretraining_objective = "masked_language_modeling"
    return AntigenLMForMaskedLM(config)


def export_runtime_files(output_dir):
    output_dir = Path(output_dir)
    repo_root = Path(__file__).resolve().parents[1]
    template_dir = repo_root / "LLM" / "AntigenLM_distilled"
    for filename in ("configuration_antigenlm.py", "modeling_antigenlm.py"):
        source = template_dir / filename
        if source.exists():
            shutil.copyfile(source, output_dir / filename)


def write_model_card(output_dir, config):
    output_dir = Path(output_dir)
    model_card = {
        "model_name": config["name"],
        "architecture_family": "ESM-C-style protein language model",
        "implementation_note": (
            "The training code uses the ESM-C architecture implementation, but parameters "
            "are randomly initialized and pre-trained as an AntigenLM MLM checkpoint."
        ),
        "parameter_initialization": "random",
        "base_pretrained_checkpoint": None,
        "pretraining_source": "from_scratch",
        "pretraining_objective": "masked_language_modeling",
        "loader": "AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)",
        "requires": ["torch", "transformers", "safetensors", "esm"],
    }
    import json

    with (output_dir / "model_card.json").open("w", encoding="utf-8") as handle:
        json.dump(model_card, handle, indent=2)
        handle.write("\n")


def main(config):
    os.environ.setdefault("MASTER_PORT", "29501")
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
        eval_accumulation_steps=config["trainer"].get("eval_accumulation_steps"),
        logging_strategy="steps",
        logging_steps=config["trainer"]["logging_steps"],
        save_strategy="steps",
        save_total_limit=config["trainer"].get("save_total_limit", 5),
        dataloader_num_workers=config["trainer"].get("dataloader_num_workers", 8),
        load_best_model_at_end=test_dataset is not None,
        metric_for_best_model=config["trainer"].get("metric_for_best_model", "eval_acc"),
        logging_dir=config._log_dir,
        fp16=config["deepspeed"].get("fp16", {}).get("enabled", False),
        deepspeed=config["deepspeed"],
        gradient_accumulation_steps=config["deepspeed"].get("gradient_accumulation_steps", 1),
        optim="adamw_torch",
        report_to="none",
        disable_tqdm=False,
        no_cuda=False,
        skip_memory_metrics=True,
    )

    model = build_antigenlm_model(
        logger=logger,
        model_args=config["model"]["args"],
        vocab_size=dataset.get_vocab_size(),
        pad_token_id=dataset.get_pad_token_id(),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params}")

    maa_metrics = MAA_metrics(
        token_with_special_list=dataset.get_token_list(),
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
    export_runtime_files(config._save_dir)
    write_model_card(config._save_dir, config)
    logger.info(f"AntigenLM ESM-C-style pretraining checkpoint saved at {config._save_dir}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    args = argparse.ArgumentParser(description="Pre-train AntigenLM with an ESM-C-style backbone")
    args.add_argument("-c", "--config", default="config/bert_pretrain_antigenlm_esmc_scratch.json", type=str)
    args.add_argument("-r", "--resume", default=None, type=str)
    args.add_argument("-d", "--device", default=None, type=str)
    args.add_argument("-local_rank", "--local_rank", default=0, type=int)

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="trainer;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="trainer;batch_size"),
    ]

    parsed_config = ConfigParser.from_args(args, options)
    main(parsed_config)
