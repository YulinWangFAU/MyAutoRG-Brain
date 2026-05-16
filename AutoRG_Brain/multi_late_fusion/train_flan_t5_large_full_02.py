# -*- coding: utf-8 -*-
"""
Flan-T5-Large Full Fine-tuning - Fixed Version

Author: Yulin Wang
Email: yulin.wang@fau.de
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
OUTPUT_DIR = f"/home/woody/iwi5/iwi5325h/flan_t5_large_full_ft_fixed_{timestamp}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Output dir:", OUTPUT_DIR)

# ========================
# Helper: parameter summary
# ========================
def get_parameter_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_ratio": 100 * trainable_params / total_params,
    }

    return summary


def save_parameter_summary(model, output_dir, model_name, tuning_type):
    summary = get_parameter_summary(model)

    text = (
        f"Model: {model_name}\n"
        f"Tuning type: {tuning_type}\n"
        f"Total parameters: {summary['total_params']:,}\n"
        f"Trainable parameters: {summary['trainable_params']:,}\n"
        f"Frozen parameters: {summary['frozen_params']:,}\n"
        f"Trainable ratio: {summary['trainable_ratio']:.4f}%\n"
    )

    print("\n========== Parameter Summary ==========")
    print(text)
    print("=======================================\n")

    with open(os.path.join(output_dir, "parameter_info.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    with open(os.path.join(output_dir, "parameter_info.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

# ========================
# Load Data
# ========================
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

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

        model_inputs["labels"] = labels["input_ids"]

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

# Save parameter information
param_summary = save_parameter_summary(
    model=model,
    output_dir=OUTPUT_DIR,
    model_name=model_name,
    tuning_type="Full fine-tuning"
)

trainable_params = param_summary["trainable_params"]
total_params = param_summary["total_params"]
frozen_params = param_summary["frozen_params"]
trainable_ratio = param_summary["trainable_ratio"]

# ========================
# Dataset Instances
# ========================
train_dataset = FusionDataset(train_data, tokenizer)
val_dataset = FusionDataset(val_data, tokenizer)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

# ========================
# Metrics
# ========================
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

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

    logging_steps=10,
    logging_first_step=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    report_to="tensorboard",

    predict_with_generate=True,
    generation_max_length=256,
    generation_num_beams=1,

    max_grad_norm=1.0,

    save_safetensors=False,
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
# Test one batch before training
# ========================
batch = next(iter(trainer.get_train_dataloader()))
batch = {k: v.to(model.device) for k, v in batch.items()}

outputs = model(**batch)
print("Initial loss:", outputs.loss.item())

with open(os.path.join(OUTPUT_DIR, "initial_check.txt"), "w", encoding="utf-8") as f:
    f.write(f"Initial loss: {outputs.loss.item()}\n")
    f.write(f"Input batch keys: {list(batch.keys())}\n")
    f.write(f"Input ids shape: {batch['input_ids'].shape}\n")
    f.write(f"Labels shape: {batch['labels'].shape}\n")
    f.write(f"Valid label tokens: {(batch['labels'] != -100).sum().item()}\n")

# ========================
# Train
# ========================
trainer.train()

# ========================
# Best checkpoint info
# ========================
best_ckpt = trainer.state.best_model_checkpoint
best_metric = trainer.state.best_metric
best_step = best_ckpt.split("-")[-1] if best_ckpt else "N/A"

print("Best checkpoint:", best_ckpt)
print("Best metric:", best_metric)
print("Best step:", best_step)

with open(os.path.join(OUTPUT_DIR, "best_result.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best checkpoint: {best_ckpt}\n")
    f.write(f"Best metric: {best_metric}\n")
    f.write(f"Best step: {best_step}\n")

# ========================
# Save final full model
# ========================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training finished.")
print("Output dir:", OUTPUT_DIR)

# ========================
# Save validation predictions
# ========================
print("Generating validation predictions...")

pred_output = trainer.predict(val_dataset)
preds = pred_output.predictions

if isinstance(preds, tuple):
    preds = preds[0]

decoded_preds = tokenizer.batch_decode(
    preds,
    skip_special_tokens=True,
)

val_predictions = []

for i, pred in enumerate(decoded_preds):
    val_predictions.append({
        "case_id": val_data[i].get("case_id", str(i)),
        "input_text": val_data[i]["input_text"],
        "target_text": val_data[i]["target_text"],
        "prediction": pred.strip(),
    })

val_pred_path = os.path.join(OUTPUT_DIR, "val_predictions.json")

with open(val_pred_path, "w", encoding="utf-8") as f:
    json.dump(val_predictions, f, indent=2, ensure_ascii=False)

print("Saved validation predictions to:", val_pred_path)

# ========================
# Save readable sample predictions
# ========================
sample_pred_path = os.path.join(OUTPUT_DIR, "sample_val_predictions.txt")

with open(sample_pred_path, "w", encoding="utf-8") as f:
    for i in range(min(5, len(val_predictions))):
        f.write("=" * 80 + "\n")
        f.write(f"Case ID: {val_predictions[i]['case_id']}\n\n")
        f.write("[PREDICTION]\n")
        f.write(val_predictions[i]["prediction"] + "\n\n")
        f.write("[GROUND TRUTH]\n")
        f.write(val_predictions[i]["target_text"] + "\n\n")

print("Saved sample predictions to:", sample_pred_path)

# ========================
# Save Training Logs and Curve
# ========================
import pandas as pd
import matplotlib.pyplot as plt

logs = pd.DataFrame(trainer.state.log_history)

if "step" not in logs.columns:
    logs["step"] = range(len(logs))

logs_path = os.path.join(OUTPUT_DIR, "training_logs.csv")
logs.to_csv(logs_path, index=False)
print("Saved training logs to:", logs_path)

train_logs = logs[logs["loss"].notna()] if "loss" in logs.columns else pd.DataFrame()
eval_logs = logs[logs["eval_loss"].notna()] if "eval_loss" in logs.columns else pd.DataFrame()

plt.figure()

if not train_logs.empty:
    plt.plot(
        train_logs["step"],
        train_logs["loss"],
        label="train_loss",
    )

if not eval_logs.empty:
    plt.plot(
        eval_logs["step"],
        eval_logs["eval_loss"],
        marker="o",
        label="val_loss",
    )

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Training Curve")

loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_curve_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved loss curve to:", loss_curve_path)

# ========================
# Save summary
# ========================
summary_path = os.path.join(OUTPUT_DIR, "run_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Flan-T5-Large Full Fine-tuning - Fixed Version\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Train path: {TRAIN_PATH}\n")
    f.write(f"Val path: {VAL_PATH}\n")
    f.write(f"Output dir: {OUTPUT_DIR}\n")
    f.write(f"Train samples: {len(train_data)}\n")
    f.write(f"Val samples: {len(val_data)}\n")
    f.write(f"Total parameters: {total_params:,}\n")
    f.write(f"Trainable parameters: {trainable_params:,}\n")
    f.write(f"Frozen parameters: {frozen_params:,}\n")
    f.write(f"Trainable ratio: {trainable_ratio:.4f}%\n")
    f.write(f"Initial loss: {outputs.loss.item()}\n")
    f.write(f"Best checkpoint: {best_ckpt}\n")
    f.write(f"Best metric: {best_metric}\n")
    f.write(f"Best step: {best_step}\n")
    f.write(f"Validation predictions: {val_pred_path}\n")
    f.write(f"Sample predictions: {sample_pred_path}\n")
    f.write(f"Training logs: {logs_path}\n")
    f.write(f"Loss curve: {loss_curve_path}\n")

print("Saved run summary to:", summary_path)
print("All done.")