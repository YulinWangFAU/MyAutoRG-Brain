# -*- coding: utf-8 -*-
"""
Created on 2026/2/20 17:22

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# !/usr/bin/env python
"""
Main training script for multimodal MRI report generation
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from data.dataset import BrainMRIDataset
from data.collator import MultimodalDataCollator
from data.utils import load_json, analyze_dataset, print_dataset_stats
from models.t5_model import BrainMRIReportGenerator
from training.trainer import Trainer


def main():
    # 配置
    config = {
        'model_name': 't5-base',
        'batch_size': 4,
        'learning_rate': 3e-5,
        'num_epochs': 20,
        'max_input_length': 512,
        'max_target_length': 256,
        'mask_prob': 0.2,
        'checkpoint_dir': 'experiments',
        'data_paths': {
            'modal_wise': 'modal_wise_finding.json',
            'global': 'global_finding.json',
            'split': 'train_val_test_case_level_split_GLI_MEN.json'
        }
    }

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 分析数据集
    stats = analyze_dataset(
        config['data_paths']['modal_wise'],
        config['data_paths']['global'],
        config['data_paths']['split']
    )
    print_dataset_stats(stats)

    # 初始化tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config['model_name'])
    special_tokens = ['<T1>', '</T1>', '<T2>', '</T2>',
                      '<FLAIR>', '</FLAIR>', '<T1C>', '</T1C>',
                      '<MISSING>']
    tokenizer.add_tokens(special_tokens)

    # 加载数据划分
    splits = load_json(config['data_paths']['split'])
    train_ids = [sid for sid in splits['train'] if sid.startswith('BraTS-GLI')]
    val_ids = [sid for sid in splits['val'] if sid.startswith('BraTS-GLI')]
    test_ids = [sid for sid in splits['test'] if sid.startswith('BraTS-GLI')]

    print(f"\nTrain samples: {len(train_ids)}")
    print(f"Val samples: {len(val_ids)}")
    print(f"Test samples: {len(test_ids)}")

    # 创建数据集
    train_dataset = BrainMRIDataset(
        subject_ids=train_ids,
        modal_wise_file=config['data_paths']['modal_wise'],
        global_file=config['data_paths']['global'],
        tokenizer=tokenizer,
        max_length=config['max_input_length'],
        target_length=config['max_target_length'],
        mode='train'
    )

    val_dataset = BrainMRIDataset(
        subject_ids=val_ids,
        modal_wise_file=config['data_paths']['modal_wise'],
        global_file=config['data_paths']['global'],
        tokenizer=tokenizer,
        max_length=config['max_input_length'],
        target_length=config['max_target_length'],
        mode='val'
    )

    # 创建data loaders
    train_collator = MultimodalDataCollator(
        tokenizer,
        mask_prob=config['mask_prob'],
        mode='train'
    )

    val_collator = MultimodalDataCollator(
        tokenizer,
        mask_prob=0.0,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=train_collator,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=val_collator,
        num_workers=2
    )

    # 初始化模型
    model = BrainMRIReportGenerator(
        model_name=config['model_name'],
        special_tokens=special_tokens
    )
    model.to(device)

    # 训练
    trainer = Trainer(model, tokenizer, device, config)

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.save_checkpoint(epoch, val_loss, is_best=True)

        # 定期保存checkpoint
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(epoch, val_loss, is_best=False)

    print("Training completed!")


if __name__ == "__main__":
    main()