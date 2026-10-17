# -*- coding: utf-8 -*-
"""
Created on 2026/2/18 11:29
t5-small
@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# train_t5_late_fusion_02.py

import argparse
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

import numpy as np
import evaluate
from transformers import EarlyStoppingCallback
from transformers import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the T5-Small text-fusion model reproducibly."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_path",
        default=(
            "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/"
            "multi_late_fusion/late_fusion_data/late_fusion_train.json"
        ),
    )
    parser.add_argument(
        "--val_path",
        default=(
            "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/"
            "multi_late_fusion/late_fusion_data/late_fusion_val.json"
        ),
    )
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


args = parse_args()
set_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ========================
# 路径
# ========================

from datetime import datetime

TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = args.output_dir or (
    f"/home/woody/iwi5/iwi5325h/t5_late_fusion_model_{timestamp}"
)

for data_path in (TRAIN_PATH, VAL_PATH):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Missing dataset file: {data_path}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
    json.dump(
        {
            "model": "t5-small",
            "seed": args.seed,
            "train_path": TRAIN_PATH,
            "val_path": VAL_PATH,
            "output_dir": OUTPUT_DIR,
            "best_model_metric": "rouge1",
        },
        f,
        indent=2,
    )

# ========================
# 读取数据
# ========================

with open(TRAIN_PATH) as f:
    train_data = json.load(f)

with open(VAL_PATH) as f:
    val_data = json.load(f)

# ========================
# Dataset 类
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
        input_text = sample["input_text"]
        target_text = sample["target_text"]

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            target_text,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels_ids = labels["input_ids"].squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

        model_inputs = {k: v.squeeze() for k, v in model_inputs.items()}
        model_inputs["labels"] = labels_ids

        return model_inputs

# ========================
# 加载模型
# ========================

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

train_dataset = FusionDataset(train_data, tokenizer)
val_dataset = FusionDataset(val_data, tokenizer)

# ========================
# ROUGE 评估
# ========================

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # 如果是 tuple，取第一个
    if isinstance(preds, tuple):
        preds = preds[0]

    # 把 preds 里非法值裁剪
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)

    # 处理 labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    return {
        "rouge1": result["rouge1"],
        "rougeL": result["rougeL"]
    }


# ========================
# 训练参数
# ========================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    fp16=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    warmup_ratio=0.1,
    report_to="tensorboard",
    predict_with_generate=True,      # ✅ 现在可以写
    generation_max_length=256,       # ✅ 现在可以写
    seed=args.seed,
    data_seed=args.seed,
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
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)




# ========================
# 开始训练
# ========================

trainer.train()
best_ckpt = trainer.state.best_model_checkpoint
best_metric = trainer.state.best_metric
best_step = best_ckpt.split("-")[-1] if best_ckpt else "N/A"

print("Best model checkpoint:", best_ckpt)
print("Best metric:", trainer.state.best_metric)
print("Best step:", best_step)

with open(os.path.join(OUTPUT_DIR, "best_result.txt"), "w") as f:
    f.write(f"Seed: {args.seed}\n")
    f.write(f"Best checkpoint: {best_ckpt}\n")
    f.write(f"Best metric: {best_metric}\n")
    f.write(f"Best step: {best_step}\n")
# 保存最佳模型
trainer.save_model(OUTPUT_DIR)

print("Training finished.")

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
plt.title("Training and Validation Loss")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

logs.to_csv(os.path.join(OUTPUT_DIR, "training_logs.csv"), index=False)
