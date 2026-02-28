# -*- coding: utf-8 -*-
"""
Created on 2026/2/28 14:20

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# flan-t5-large
# -*- coding: utf-8 -*-
"""
flan-t5-large + LoRA (add PullLoss)

Author: Yulin Wang
"""

import os
import json
import torch
import numpy as np
import evaluate
from datetime import datetime
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from typing import List, Dict
# ========================
# Reproducibility
# ========================
set_seed(42)

# ========================
# Paths
# ========================
TRAIN_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/late_fusion_data/late_fusion_PullLoss_train.json"
VAL_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/late_fusion_data/late_fusion_PullLoss_val.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/home/woody/iwi5/iwi5325h/flan_t5_large_lora_PullLoss_{timestamp}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# Load Data
# ========================
with open(TRAIN_PATH) as f:
    train_data = json.load(f)

with open(VAL_PATH) as f:
    val_data = json.load(f)

# ========================
# Dataset
# ========================

class FusionDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len=512, max_output_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 主输入
        model_inputs = self.tokenizer(
            sample["input_text"],
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )

        # target
        labels = self.tokenizer(
            sample["target_text"],
            max_length=self.max_output_len,
            truncation=True,
            padding=False,
        )

        labels_ids = [
            (l if l != self.tokenizer.pad_token_id else -100)
            for l in labels["input_ids"]
        ]

        model_inputs["labels"] = labels_ids

        # 🔥 关键：预 tokenize 每个 modal
        for modal_key in ["t1_text", "t2_text", "flair_text", "t1c_text"]:

            modal_tok = self.tokenizer(
                sample[modal_key],
                max_length=256,
                truncation=True,
                padding=False,
            )

            model_inputs[f"{modal_key}_ids"] = modal_tok["input_ids"]
            model_inputs[f"{modal_key}_mask"] = modal_tok["attention_mask"]

        return model_inputs

# ========================
# Custom Collator
# ========================
@dataclass
class MultiModalCollator:
    tokenizer: T5Tokenizer

    def __call__(self, features: List[Dict]):

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )

        for modal in ["t1_text", "t2_text", "flair_text", "t1c_text"]:
            ids_key = f"{modal}_ids"
            mask_key = f"{modal}_mask"

            modal_features = [
                {
                    "input_ids": f[ids_key],
                    "attention_mask": f[mask_key]
                }
                for f in features
            ]

            modal_batch = self.tokenizer.pad(
                modal_features,
                padding=True,
                return_tensors="pt"
            )

            batch[ids_key] = modal_batch["input_ids"]
            batch[mask_key] = modal_batch["attention_mask"]

        return batch


# ========================
# Model & Tokenizer
# ========================

model_name = "google/flan-t5-large"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ========================
# LoRA Config
# ========================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
model.print_trainable_parameters()

# ========================
# Dataset Instances
# ========================

train_dataset = FusionDataset(train_data, tokenizer)
val_dataset = FusionDataset(val_data, tokenizer)

data_collator = MultiModalCollator(tokenizer)

# ========================
# Metrics
# ========================

rouge = evaluate.load("rouge")
# bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    # bert_result = bertscore.compute(
    #     predictions=decoded_preds,
    #     references=decoded_labels,
    #     lang="en",
    # )

    return {
        "rouge1": rouge_result["rouge1"],
        "rougeL": rouge_result["rougeL"],
        # "bertscore_f1": np.mean(bert_result["f1"]),
    }

# ========================
# Trainer with Multi-Positive Contrastive
# ========================
class MultiModalTrainer(Seq2SeqTrainer):
    def masked_mean(self, hidden, mask):
        mask = mask.unsqueeze(-1)  # [B, L, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts
    def compute_loss(self, model, inputs, return_outputs=False):

        device = model.device
        tau = 0.1
        lambda_contrast = 0.05

        # ===== LM Loss =====
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        lm_loss = outputs.loss

        encoder = model.get_encoder()

        # ===== Global embedding =====
        global_enc = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        z_global = self.masked_mean(
            global_enc.last_hidden_state,
            inputs["attention_mask"]
        )

        z_global = F.normalize(z_global, dim=-1)

        # ===== Modal embeddings =====
        modal_list = ["t1_text", "t2_text", "flair_text", "t1c_text"]
        modal_embeds = []

        for modal in modal_list:
            enc_out = encoder(
                input_ids=inputs[f"{modal}_ids"],
                attention_mask=inputs[f"{modal}_mask"]
            )
            z_modal = self.masked_mean(
                enc_out.last_hidden_state,
                inputs[f"{modal}_mask"]
            )
            z_modal = F.normalize(z_modal, dim=-1)
            modal_embeds.append(z_modal)

        modal_embeds = torch.stack(modal_embeds, dim=1)
        B, M, D = modal_embeds.shape
        modal_flat = modal_embeds.view(B * M, D)

        # ===== Similarity matrix =====
        logits_g2m = torch.matmul(z_global, modal_flat.T) / tau  # [B , B*M]

        # ===== Positive mask (global → modal) =====
        mask = torch.zeros_like(logits_g2m, dtype=torch.bool)

        indices = torch.arange(B, device=device)
        mask = torch.zeros(B, B * M, device=device, dtype=torch.bool)
        for i in range(B):
            mask[i, i * M:(i + 1) * M] = True

        logits_pos = logits_g2m.masked_fill(~mask, float('-inf'))

        numerator = torch.logsumexp(logits_pos, dim=1)
        denominator = torch.logsumexp(logits_g2m, dim=1)

        loss_g2m = -(numerator - denominator).mean()

        # ===== modal → global =====
        logits_m2g = torch.matmul(modal_flat, z_global.T) / tau  # [B*M , B]

        labels = torch.arange(B).to(device).repeat_interleave(M)

        loss_m2g = F.cross_entropy(logits_m2g, labels)

        # ===== Final contrastive loss =====
        contrastive_loss = (loss_g2m + loss_m2g) / 2

        total_loss = lm_loss + lambda_contrast * contrastive_loss

        self.log({
            "lm_loss": lm_loss.detach().cpu().item(),
            "contrastive_loss": contrastive_loss.detach().cpu().item()
        })

        return (total_loss, outputs) if return_outputs else total_loss

# ========================
# Training Arguments
# ========================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # 🔥 more stable for large model
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    weight_decay=0.01,
    # warmup_ratio=0.1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    #bf16=True,
    fp16=True,
    logging_steps=20,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    report_to="tensorboard",
    predict_with_generate=True,
    generation_max_length=256,
    max_grad_norm=1.0,
)

# ========================
# Trainer
# ========================

trainer = MultiModalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ========================
# Train
# ========================
trainer.train()

best_ckpt = trainer.state.best_model_checkpoint
best_metric = trainer.state.best_metric
best_step = best_ckpt.split("-")[-1] if best_ckpt else "N/A"

print("Best checkpoint:", best_ckpt)
print("Best metric:", best_metric)
print("Best step:", best_step)

with open(os.path.join(OUTPUT_DIR, "best_result.txt"), "w") as f:
    f.write(f"Best checkpoint: {best_ckpt}\n")
    f.write(f"Best metric: {best_metric}\n")
    f.write(f"Best step: {best_step}\n")

# Save final model
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished.")

# ========================
# Save Training Curve
# ========================

import pandas as pd
import matplotlib.pyplot as plt

logs = pd.DataFrame(trainer.state.log_history)

if "step" not in logs.columns:
    logs["step"] = range(len(logs))

train_logs = logs[logs["loss"].notna()]
eval_logs = logs[logs["eval_loss"].notna()]

plt.figure()
plt.plot(train_logs["step"], train_logs["loss"], label="train_loss")
plt.plot(eval_logs["step"], eval_logs["eval_loss"], marker="o", label="val_loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Training Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

logs.to_csv(os.path.join(OUTPUT_DIR, "training_logs.csv"), index=False)
