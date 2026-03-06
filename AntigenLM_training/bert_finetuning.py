import os
import argparse
import collections
import torch
import numpy as np
import transformers
from os.path import join

import data.bert_finetuning_pathogen_dataset as module_data  # 适用于病原微生物的微调数据集
import model.loss as module_loss
import model.metric as module_metric
import model.bert_pathogen as module_arch  # 适用于病原微生物的模型
from trainer.bert_finetuning_pathogen_trainer import BERTPathogenTrainer as Trainer
from parse_config import ConfigParser

def main(config):
    logger = config.get_logger('train')
    
    # 固定随机种子，确保可复现性
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # 加载数据集
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_dataset(valid=True)
    test_data_loader = data_loader.split_dataset(valid=False, test=True)
    logger.info('Number of samples in train: {}, valid: {}, and test: {}.'.format(
        data_loader.sampler.__len__(), 
        valid_data_loader.sampler.__len__(), 
        test_data_loader.sampler.__len__()
    ))
    
    # 加载预训练的微生物语言模型
    pretrained_model_path = config['pretrained_model_path']
    model = config.init_obj('arch', module_arch, pretrained_model_path=pretrained_model_path)
    
    # 选择需要优化的参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # 设置优化器
    optimizer = torch.optim.AdamW(trainable_params, 
                                  lr=config['optimizer']['args']['lr'], 
                                  weight_decay=config['optimizer']['args']['weight_decay'])
    
    # 获取损失函数和评估指标
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    # 设置学习率调度器
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    # 初始化训练器
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)
    
    # 开始训练
    trainer.train()
    
    # 测试
    logger = config.get_logger('test')
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading best checkpoint from: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    
    # 保存微调后的模型
    finetuned_model_dir = join(config.save_dir, 'finetuned_pathogen_model')
    logger.info(f'Saving fine-tuned model to {finetuned_model_dir}')
    os.makedirs(finetuned_model_dir, exist_ok=True)
    model.save_pretrained(finetuned_model_dir)
    
    # 运行测试
    test_output = trainer.test()
    logger.info(test_output)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Fine-tune Pathogen Model')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='Path to config file (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='Path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='Indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='Local rank for distributed training')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    
    config = ConfigParser.from_args(args, options)
    main(config)
