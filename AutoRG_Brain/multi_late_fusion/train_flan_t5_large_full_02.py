# -*- coding: utf-8 -*-
"""
Created on 2026/5/13 20:28

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Flan-T5-Large Full Fine-tuning
Author: Yulin Wang
"""

import os
import json
import torch
import numpy as np
import evaluate
from datetime import datetime
from torch.utils.data import Dataset

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)

# ========================
# Reproducibility
# ========================
set_seed(42)

# ========================
# Paths
# ========================
TRAIN_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/late_fusion_data/late_fusion_train.json"
VAL_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/late_fusion_data/late_fusion_val.json"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"/home/woody/iwi5/iwi5325h/flan_t5_large_full_ft_{timestamp}"

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

        model_inputs = self.tokenizer(
            sample["input_text"],
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )

        labels = self.tokenizer(
            sample["target_text"],
            max_length=self.max_output_len,
            truncation=True,
            padding=False,
        )

        labels_ids = labels["input_ids"]

        labels_ids = [
            l if l != self.tokenizer.pad_token_id else -100
            for l in labels_ids
        ]

        model_inputs["labels"] = labels_ids
        return model_inputs

# ========================
# Model & Tokenizer
# ========================
model_name = "google/flan-t5-large"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Full fine-tuning: all parameters are trainable
for param in model.parameters():
    param.requires_grad = True

# Reduce memory usage
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

# ========================
# Dataset Instances
# ========================
train_dataset = FusionDataset(train_data, tokenizer)
val_dataset = FusionDataset(val_data, tokenizer)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

# ========================
# Metrics
# ========================
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "rouge1": rouge_result["rouge1"],
        "rougeL": rouge_result["rougeL"],
    }

# ========================
# Training Arguments
# ========================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,

    num_train_epochs=20,
    weight_decay=0.01,

    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,

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
trainer = Seq2SeqTrainer(
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
# Test one batch
# ========================
batch = next(iter(trainer.get_train_dataloader()))
batch = {k: v.to(model.device) for k, v in batch.items()}

outputs = model(**batch)
print("Initial loss:", outputs.loss.item())

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

# Save final full model
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished.")
print("Output dir:", OUTPUT_DIR)

# ========================
# Save Training Curve
# ========================
import pandas as pd
import matplotlib.pyplot as plt

logs = pd.DataFrame(trainer.state.log_history)

if "step" not in logs.columns:
    logs["step"] = range(len(logs))

train_logs = logs[logs["loss"].notna()] if "loss" in logs.columns else pd.DataFrame()
eval_logs = logs[logs["eval_loss"].notna()] if "eval_loss" in logs.columns else pd.DataFrame()

plt.figure()

if not train_logs.empty:
    plt.plot(train_logs["step"], train_logs["loss"], label="train_loss")

if not eval_logs.empty:
    plt.plot(eval_logs["step"], eval_logs["eval_loss"], marker="o", label="val_loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Training Curve")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

logs.to_csv(os.path.join(OUTPUT_DIR, "training_logs.csv"), index=False)