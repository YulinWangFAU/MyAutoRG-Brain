# -*- coding: utf-8 -*-
"""
Created on 2026/2/20 17:17

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
"""
Trainer for multimodal MRI report generation
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-5)
        )

        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint_dir = self.config.get('checkpoint_dir', 'experiments')
        if is_best:
            save_path = os.path.join(checkpoint_dir, 'best_model')
        else:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}')

        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # 保存训练状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(save_path, 'training_state.pt'))

        print(f"Saved checkpoint to {save_path}")