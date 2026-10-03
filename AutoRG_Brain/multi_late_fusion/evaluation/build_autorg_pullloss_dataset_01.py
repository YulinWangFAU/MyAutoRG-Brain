# -*- coding: utf-8 -*-
"""
Created on 2026/3/13 17:42

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Build Late Fusion dataset using AutoRG single-modality reports
把 AutoRG-Brain 为每个 MRI 序列生成的单模态报告，组合成一个文本输入 prompt，并以人工标注的全局报告作为 target，构造用于训练、验证和测试 Late Fusion 文本模型的数据集。
Author: Yulin Wang
"""

import json
import os

# ===============================
# 路径
# ===============================

SPLIT_PATH = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json"

GLI_GLOBAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_GLI/global_finding.json"
MEN_GLOBAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_MEN/global_finding.json"

# AutoRG 单模态报告
AUTORG_TRAIN = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/Autorg_output/autorg_predictions/autorg_predictions_train.json"
AUTORG_VAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/Autorg_output/autorg_predictions/autorg_predictions_val.json"
AUTORG_TEST = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/Autorg_output/autorg_predictions/autorg_predictions_test.json"

OUTPUT_DIR = "late_fusion_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# 读取 split
# ===============================

with open(SPLIT_PATH) as f:
    split = json.load(f)

train_cases = split["train"]
val_cases = split["val"]
test_cases = split["test"]


# ===============================
# 读取 global report
# ===============================

with open(GLI_GLOBAL) as f:
    gli_global = json.load(f)

with open(MEN_GLOBAL) as f:
    men_global = json.load(f)

global_dict = {}
global_dict.update(gli_global)
global_dict.update(men_global)


# ===============================
# 读取 AutoRG 单模态报告
# ===============================

modal_dict = {}

with open(AUTORG_TRAIN) as f:
    modal_dict.update(json.load(f))

with open(AUTORG_VAL) as f:
    modal_dict.update(json.load(f))

with open(AUTORG_TEST) as f:
    modal_dict.update(json.load(f))

print("Total AutoRG modality reports:", len(modal_dict))


# ===============================
# 构造 prompt
# ===============================

def build_input(case_id):

    t1 = modal_dict.get(case_id + "-t1n", "")
    t2 = modal_dict.get(case_id + "-t2w", "")
    flair = modal_dict.get(case_id + "-t2f", "")
    t1ce = modal_dict.get(case_id + "-t1c", "")

    prompt = f"""
You are a radiology expert.

Below are modality-specific MRI findings for the same patient.

[T1-weighted Imaging]
{t1}

[T2-weighted Imaging]
{t2}

[FLAIR Imaging]
{flair}

[T1CE Imaging]
{t1ce}

Please integrate the above findings into a comprehensive radiology report for this patient.
""".strip()

    return prompt


# ===============================
# 构造 dataset
# ===============================

def build_dataset(case_list):

    dataset = []

    missing_t1 = 0
    missing_t2 = 0
    missing_flair = 0
    missing_t1c = 0

    for case_id in case_list:

        if case_id not in global_dict:
            continue

        t1 = modal_dict.get(case_id + "-t1n", "")
        t2 = modal_dict.get(case_id + "-t2w", "")
        flair = modal_dict.get(case_id + "-t2f", "")
        t1ce = modal_dict.get(case_id + "-t1c", "")

        if t1 == "":
            missing_t1 += 1
        if t2 == "":
            missing_t2 += 1
        if flair == "":
            missing_flair += 1
        if t1ce == "":
            missing_t1c += 1

        input_text = build_input(case_id)
        target_text = global_dict[case_id]

        dataset.append({
            "case_id": case_id,
            "input_text": input_text,
            "target_text": target_text,
            "t1_text": t1,
            "t2_text": t2,
            "flair_text": flair,
            "t1c_text": t1ce
        })

    print("Missing T1:", missing_t1)
    print("Missing T2:", missing_t2)
    print("Missing FLAIR:", missing_flair)
    print("Missing T1CE:", missing_t1c)

    return dataset


# ===============================
# 构造 train/val/test
# ===============================

train_data = build_dataset(train_cases)
val_data = build_dataset(val_cases)
test_data = build_dataset(test_cases)


# ===============================
# 保存
# ===============================

train_path = os.path.join(OUTPUT_DIR, "late_fusion_PullLoss_autorgmodal_train.json")
val_path = os.path.join(OUTPUT_DIR, "late_fusion_PullLoss_autorgmodal_val.json")
test_path = os.path.join(OUTPUT_DIR, "late_fusion_PullLoss_autorgmodal_test.json")

with open(train_path, "w") as f:
    json.dump(train_data, f, indent=2)

with open(val_path, "w") as f:
    json.dump(val_data, f, indent=2)

with open(test_path, "w") as f:
    json.dump(test_data, f, indent=2)


print("\nDataset created!")
print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))

print("\nSaved to:")
print(train_path)
print(val_path)
print(test_path)