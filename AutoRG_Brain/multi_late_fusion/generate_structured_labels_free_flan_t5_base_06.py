# -*- coding: utf-8 -*-
"""
Created on 2026/2/20 12:25

Flan-T5-base 是：
没有专门训练医学 IE
但足够做弱监督标注

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# -*- coding: utf-8 -*-
"""
Generate structured labels for modal-wise findings
Single-sequence structured extraction
"""

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# 路径
# =========================
SPLIT_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json"

GLI_MODAL = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/RadGenome-Brain_MRI/BraTS_GLI/modal_wise_finding.json"
MEN_MODAL = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/RadGenome-Brain_MRI/BraTS_MEN/modal_wise_finding.json"

OUTPUT_DIR = "structured_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 加载 split
# =========================
with open(SPLIT_PATH) as f:
    split = json.load(f)

train_cases = split["train"]
val_cases = split["val"]
test_cases = split["test"]

# =========================
# 加载 modal-wise 报告
# =========================
with open(GLI_MODAL) as f:
    gli_modal = json.load(f)

with open(MEN_MODAL) as f:
    men_modal = json.load(f)

modal_dict = {}
modal_dict.update(gli_modal)
modal_dict.update(men_modal)

# =========================
# 加载模型（免费）
# =========================
MODEL_NAME = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# =========================
# prompt 构造
# =========================
def build_prompt(report_text):
    return f"""
Extract structured radiological findings.

Fields:
location:
sequence:
signal:
enhancement:
edema:
size:
mass_effect:
midline_shift:

Rules:
- Always output all fields.
- Use NONE if not mentioned.
- Do not add explanations.

Report:
{report_text}
"""

def extract_structured(report_text):
    prompt = build_prompt(report_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            num_beams=4
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

# =========================
# 构造 dataset
# =========================
def build_dataset(case_list):
    dataset = []

    for case_id in tqdm(case_list):
        for suffix, modality in [
            ("-t1n", "T1"),
            ("-t2w", "T2"),
            ("-t2f", "FLAIR"),
            ("-t1c", "T1CE"),
        ]:
            key = case_id + suffix

            if key not in modal_dict:
                continue

            report_text = modal_dict[key]

            structured_output = extract_structured(report_text)

            dataset.append({
                "case_id": case_id,
                "modality": modality,
                "input_text": report_text,
                "structured_output": structured_output
            })

    return dataset

# =========================
# 生成数据
# =========================
print("Generating structured train...")
train_data = build_dataset(train_cases)

print("Generating structured val...")
val_data = build_dataset(val_cases)

print("Generating structured test...")
test_data = build_dataset(test_cases)

# =========================
# 保存
# =========================
with open(os.path.join(OUTPUT_DIR, "structured_train.json"), "w") as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "structured_val.json"), "w") as f:
    json.dump(val_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "structured_test.json"), "w") as f:
    json.dump(test_data, f, indent=2)

print("Done!")
print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))
