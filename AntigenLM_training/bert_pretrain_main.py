#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import collections
import torch
import numpy as np
import deepspeed
import os
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
# ====== 本地模块 ====== #
import data.bert_pretrain_maa_dataset as module_data
from data.utility import DatasetSplit
from model.bert_pretrain import get_bert_model
from model.metric import MAA_metrics
from parse_config import ConfigParser
import os

os.environ["MASTER_PORT"] = "29501"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# # 如果是GPU，可改成 nccl，如果是NPU通常是hccl
deepspeed.init_distributed('nccl')


def main(config):
    """
    主函数：使用 Hugging Face 的 Trainer + DeepSpeed 进行训练
    """
    # 再次初始化分布式，保证环境就绪
    deepspeed.init_distributed()
    logger = config.get_logger('train')

    # 1. 设置随机种子（在多卡场景下也保证可复现）
    seed = config['dataset']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 如果NPU，可用 torch.npu.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # 2. 加载/划分数据集
    holdout = config['dataset']['args']['test_split']
    config['dataset']['args']['config'] = config
    config['dataset']['args']['logger'] = logger

    dataset = config.init_obj('dataset', module_data)
    full_train_dataset = dataset.get_dataset()
 
    test_dataset = None
    if holdout is not None:
        assert 0.0 < holdout < 1.0, "holdout 必须是介于 0 和 1 之间"
        test_dataset = DatasetSplit(
            logger=logger,
            full_dataset=full_train_dataset,
            split="test",
            valid=0,
            test=holdout
        )
        train_dataset = DatasetSplit(
            logger=logger,
            full_dataset=full_train_dataset,
            split="train",
            valid=0,
            test=holdout
        )
    else:
        train_dataset = full_train_dataset


    # 3. 分词器 & 数据整理器
    tokenizer = dataset.get_tokenizer()
    print("填充标记ID:", tokenizer.pad_token_id)

    # 如果 full_train_dataset 是通过 get_dataset() 得到的 SelfSupervisedDataset 实例
    print("每个样本的 token 数量:", full_train_dataset.max_len)
    total_tokens = len(full_train_dataset) * full_train_dataset.max_len
    print("数据集中所有 token 的总数（包括填充）:", total_tokens)

   
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 4. 构建 TrainingArguments
    training_args = TrainingArguments(
        output_dir=config['trainer']['save_dir'],  # 适配 save_dir
        overwrite_output_dir=True,
        num_train_epochs=config['trainer']['epochs'],
        per_device_train_batch_size=config['trainer']['batch_size'],
        per_device_eval_batch_size=int(config['trainer']['batch_size']),  # 适配评估批量大小
        learning_rate=config['trainer']['lr'],
        warmup_ratio=config['trainer']['warmup'],
        weight_decay=config['trainer']['weight_decay'],  # 适配权重衰减
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
        metric_for_best_model='acc',  # 确保与你的评估指标匹配
        logging_dir=config._log_dir,  # 适配日志路径
        fp16=config['deepspeed'].get("fp16", {}).get("enabled", False),  # 适配 FP16
        deepspeed=config["deepspeed"],  # 关键：直接传递 Deepspeed 配置
        gradient_accumulation_steps=config['deepspeed'].get("gradient_accumulation_steps", 1),  # 适配梯度累积
        optim="adamw_torch" if config['deepspeed']['optimizer']['type'].lower() == "adamw" else "adam",  # 适配优化器
        report_to="none",  # 关闭默认的 WandB/MLflow 记录
        disable_tqdm=False,
        no_cuda=False,  # 适配 GPU 训练
        skip_memory_metrics=True,
    )

    # 5. 初始化模型
    vocab_size = dataset.get_vocab_size()
    pad_token_id = dataset.get_pad_token_id()
    logger.info(f'词汇表大小: {vocab_size}, 填充标记ID: {pad_token_id}')

    model = get_bert_model(
        logger=logger,
        bert_variant=config['model']['bert'],
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        **config['model']['args']
    )
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(np.prod(p.size()) for p in trainable_params)
    logger.info(f'可训练参数数量: {total_params}')

    # 6. 自定义评价指标
    token_with_special_list = dataset.get_token_list()
    print(token_with_special_list)
    maa_metrics = MAA_metrics(
        token_with_special_list=token_with_special_list,
        blosum_dir=config['metrics']['blosum_dir'],
        blosum=config['metrics']['blosum']
    )

    # 7. 构建 Trainer
    # - compute_metrics 接受一个包含 'predictions', 'label_ids' 的字典
    # - 根据需要实现一个包装，调用 maa_metrics.compute_metrics()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=maa_metrics.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        # 可选：早停，若连续5次指标无提升则停止
    )

    # if config.resume is not None:
    #     # 获取当前进程的 local_rank
    #     local_rank = int(os.environ.get("LOCAL_RANK", 0))

    #     # 1) 先把 resume 与 rankN 拼接成一个目录名（字符串拼接而非 os.path.join）
    #     #    例如 config.resume = "0114_014020_", local_rank=2
    #     #    那么 rank_dir 就是 "0114_014020_rank2"
    #     rank_dir = f"{config.resume}rank{local_rank}"

    #     # 2) 再用 os.path.join 追加 checkpoint-115000
    #     #    这样最终得到 "0114_014020_rank2/checkpoint-30000"
    #     rank_resume_dir = os.path.join(rank_dir, "checkpoint-145000")
    #     print(rank_resume_dir)
    #     # 检查路径是否存在
    #     if not os.path.exists(rank_resume_dir):
    #         raise FileNotFoundError(
    #             f"Checkpoint directory for rank {local_rank} does not exist: {rank_resume_dir}"
    #         )
        
    #     # 日志打印恢复路径
    #     logger.info(f"Resuming training for rank {local_rank} from checkpoint: {rank_resume_dir}")
        
    #     # 从当前 rank 的目录恢复
    #     trainer.train(resume_from_checkpoint=rank_resume_dir)
    # else:
    if config.resume is not None:
        trainer.train(resume_from_checkpoint=config.resume)
    else:
        trainer.train()

    # 9. 训练结束后保存模型（HF Trainer 会自动使用DeepSpeed保存ZeRO权重）
    trainer.save_model(config._save_dir)
    logger.info(f"Model checkpoint saved at {config._save_dir}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='DeepSpeed BERT Pretrain Example with HF Trainer')

    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='配置文件路径 (默认: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='最新检查点路径 (默认: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='启用的GPU索引 (默认: all)')
    # DeepSpeed 多卡训练常用的 local_rank 参数，需要保留
    args.add_argument('-local_rank', '--local_rank', default=0, type=int,
                      help='多GPU/多进程训练的本地进程索引')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='trainer;batch_size')
    ]
    

    config = ConfigParser.from_args(args, options)

    main(config)
