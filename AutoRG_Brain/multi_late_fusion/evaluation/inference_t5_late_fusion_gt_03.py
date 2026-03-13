# -*- coding: utf-8 -*-
"""
Created on 2026/3/13 16:30

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Created on 2026/2/18 15:55

先只生成 prediction 文件
之后再写一个单独 metrics 脚本

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# inference_generate_predictions_t5_late_fusion.py

import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from transformers import set_seed

# =========================
# 固定随机种子（保证可复现）
# =========================
set_seed(42)

# =========================
# 修改路径
# =========================

CHECKPOINT_PATH = "/home/woody/iwi5/iwi5325h/t5_late_fusion_model_20260218_151904/checkpoint-283"
TEST_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/evaluation/late_fusion_test_autorg_modal.json"
OUTPUT_PRED_PATH = "/home/woody/iwi5/iwi5325h/t5_late_fusion_model_20260218_151904/test_predictions_late_fusion_autorgmodal.json"
# test_predictions_late_fusion_inputAutoRGmodal.json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =========================
# 加载 tokenizer & model
# =========================

tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_PATH)
model = T5ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH)
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
# 推理（只生成，不算指标）
# =========================

for sample in tqdm(test_data):

    input_text = sample["input_text"]
    reference = sample["target_text"]

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
            do_sample=False   # 关键：保证 deterministic
        )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results_to_save.append({
        "case_id": sample["case_id"],
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
