import os
import torch
import argparse
import collections
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed , EarlyStoppingCallback
)

import data.bert_finetuning_pathogen_dataset_second as module_data
import deepspeed
from model.metric import MAA_metrics
from parse_config import ConfigParser
from data.utility import StructureAwareDataCollatorForLanguageModeling

# 设置通信端口
os.environ["MASTER_PORT"] = "29501"
deepspeed.init_distributed('nccl')


def build_model(pretrained_model_path):
    # 全参数微调：直接加载预训练模型
    model = AutoModelForMaskedLM.from_pretrained(pretrained_model_path)
    return model


def main(config):
    logger = config.get_logger("train")

    # 固定随机种子
    seed = config['dataset']['args']['seed']
    set_seed(seed)

    # 加载 tokenizer 和模型
    pretrained_model_path = config['pretrained_model_path']
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = build_model(pretrained_model_path)

    # 构建数据集加载器
    config['dataset']['args']['logger'] = logger
    config['dataset']['args']['tokenizer'] = tokenizer
    dataset_obj = config.init_obj('dataset', module_data)
    train_dataset = dataset_obj.get_dataset('train')
    print(">>> DATASET SAMPLE KEYS:", train_dataset[0].keys())
    print(">>> DATASET SAMPLE:", {k: (v.shape if isinstance(v, torch.Tensor) else v) 
                               for k, v in train_dataset[0].items()})
    eval_dataset = dataset_obj.get_dataset('test')

    # 数据整理器
    data_collator = StructureAwareDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config['dataset']['args']['mlm_probability'],
        logger=logger,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config['trainer']['save_dir'],
        overwrite_output_dir=True,
        num_train_epochs=config['trainer']['epochs'],
        per_device_train_batch_size=config['trainer']['batch_size'],
        per_device_eval_batch_size=int(config['trainer']['batch_size']),
        learning_rate=config['trainer']['lr'],
        warmup_ratio=config['trainer']['warmup'],
        weight_decay=config['trainer']['weight_decay'],
        evaluation_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        eval_accumulation_steps=config['trainer'].get('eval_accumulation_steps', None),
        logging_strategy="steps",
        logging_steps=config['trainer']['logging_steps'],
        save_strategy="steps",
        save_total_limit=5,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        metric_for_best_model='acc',
        logging_dir=config._log_dir,
        fp16=config['deepspeed'].get("fp16", {}).get("enabled", False),
        deepspeed=config["deepspeed"],
        gradient_accumulation_steps=config['deepspeed'].get("gradient_accumulation_steps", 1),
        optim="adamw_torch",
        report_to="none",
        disable_tqdm=False,
        no_cuda=False,
        skip_memory_metrics=True,
        remove_unused_columns=False,
    )

    # 自定义评价指标
    token_with_special_list = dataset_obj.get_token_list()
    maa_metrics = MAA_metrics(
        token_with_special_list=token_with_special_list,
        blosum_dir=config['metrics']['blosum_dir'],
        blosum=config['metrics']['blosum'])

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=maa_metrics.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]    )

    # 启动训练
    if config.resume is not None:
        trainer.train(resume_from_checkpoint=config.resume)
    else:
        trainer.train()

    # 保存最终模型
    trainer.save_model(config._save_dir)
    logger.info(f"Model checkpoint saved at {config._save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full Fine-tuning for microLM')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='Path to config file')
    parser.add_argument('-r', '--resume', default=None, type=str, help='Path to checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='GPU device ids')
    parser.add_argument('-local_rank', '--local_rank', default=None, type=str, help='Local rank for DDP')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(parser, options)
    main(config)
