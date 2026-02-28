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

        model_inputs = self.tokenizer(
            sample["input_text"],
            max_length=self.max_input_len,
            truncation=True,
            padding=False,  # 🔥 dynamic padding
        )

        labels = self.tokenizer(
            sample["target_text"],
            max_length=self.max_output_len,
            truncation=True,
            padding=False,
        )

        labels_ids = labels["input_ids"]

        # 🔥 关键：把 padding token 替换成 -100
        labels_ids = [
            (l if l != self.tokenizer.pad_token_id else -100)
            for l in labels_ids
        ]

        model_inputs["labels"] = labels_ids

        # 🔥 新增：保存原始文本用于 pull loss
        model_inputs["t1_text"] = sample["t1_text"]
        model_inputs["t2_text"] = sample["t2_text"]
        model_inputs["flair_text"] = sample["flair_text"]
        model_inputs["t1c_text"] = sample["t1c_text"]
        model_inputs["target_text_raw"] = sample["target_text"]

        return model_inputs


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
    target_modules=["q", "v"],  # Verified for T5
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

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
# bertscore = evaluate.load("bertscore")

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
# Training Arguments
# ========================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # 🔥 more stable for large model
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    weight_decay=0.01,
    # warmup_ratio=0.1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    fp16=False,
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
class MultiModalTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        # 🔥 先取出模态文本
        t1_text = inputs.pop("t1_text")
        t2_text = inputs.pop("t2_text")
        flair_text = inputs.pop("flair_text")
        t1c_text = inputs.pop("t1c_text")
        target_text_raw = inputs.pop("target_text_raw")

        # ===== LM LOSS =====
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # ===== MULTI-MODAL PULL LOSS =====

        device = model.device

        modal_texts = [t1_text, t2_text, flair_text, t1c_text]
        modal_embeds = []

        for modal in modal_texts:
            tokenized = tokenizer(
                modal,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            modal_output = model.base_model.encoder(**tokenized)
            modal_embed = modal_output.last_hidden_state.mean(dim=1)
            modal_embeds.append(modal_embed)

        modal_embeds = torch.stack(modal_embeds, dim=1)
        modal_mean = modal_embeds.mean(dim=1)

        # encode global target
        global_tok = tokenizer(
            target_text_raw,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        global_output = model.base_model.encoder(**global_tok)
        global_embed = global_output.last_hidden_state.mean(dim=1)

        # cosine pull loss
        pull_loss = 0
        for i in range(4):
            pull_loss += 1 - F.cosine_similarity(
                global_embed,
                modal_embeds[:, i, :],
                dim=-1
            ).mean()

        pull_loss = pull_loss / 4

        # ===== TOTAL LOSS =====
        total_loss = lm_loss + 0.1 * pull_loss

        return (total_loss, outputs) if return_outputs else total_loss

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
batch = next(iter(trainer.get_train_dataloader()))
outputs = model(**batch)
print("Initial loss:", outputs.loss)

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
