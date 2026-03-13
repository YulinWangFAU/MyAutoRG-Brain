# -*- coding: utf-8 -*-
"""
Created on 2026/3/13 17:30

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Created on 2026/2/28 23:13

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Inference script for flan-t5-large + LoRA (late fusion + contrastive)

@author: Yulin Wang
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
from peft import PeftModel

# =========================
# 固定随机种子
# =========================
set_seed(42)

# =========================
# 路径（改成你这次训练的 best checkpoint）
# =========================

BASE_MODEL = "google/flan-t5-large"

LORA_CHECKPOINT = "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_PullLoss_20260228_164119/checkpoint-603"

TEST_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/late_fusion_data/late_fusion_PullLoss_autorgmodal_test.json"

OUTPUT_PRED_PATH = "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_PullLoss_20260228_164119/late_fusion_PullLoss_autorgmodal_test_predictions.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================
# 加载 tokenizer
# =========================

tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)

# =========================
# 加载 base model + LoRA
# =========================

base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
base_model.config.use_cache = True   # 推理阶段可以开

model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)

model.to(DEVICE)
model.eval()

# =========================
# 读取 test 数据
# =========================

with open(TEST_PATH) as f:
    test_data = json.load(f)

print("Number of test samples:", len(test_data))

results_to_save = []

# =========================
# 推理
# =========================

for sample in tqdm(test_data):

    input_text = sample["input_text"]
    reference = sample.get("target_text", "")

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )

    # 🔥 安全 decode（防止非法 token）
    output_ids = outputs[0].cpu().numpy()
    output_ids = np.where(
        (output_ids >= 0) & (output_ids < tokenizer.vocab_size),
        output_ids,
        tokenizer.pad_token_id
    )

    prediction = tokenizer.decode(output_ids, skip_special_tokens=True)

    results_to_save.append({
        "case_id": sample.get("case_id", ""),
        "input": input_text,
        "prediction": prediction,
        "reference": reference
    })

# =========================
# 保存 prediction 文件
# =========================

with open(OUTPUT_PRED_PATH, "w") as f:
    json.dump(results_to_save, f, indent=2)

print("\nPrediction file saved to:", OUTPUT_PRED_PATH)
print("Done.")