# -*- coding: utf-8 -*-
"""
Created on 2026/2/28 14:29

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Created on 2026/2/17 23:59

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import json
import os

# ======== 路径修改成你的 ========
SPLIT_PATH = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json"

GLI_MODAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_GLI/modal_wise_finding.json"
GLI_GLOBAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_GLI/global_finding.json"

MEN_MODAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_MEN/modal_wise_finding.json"
MEN_GLOBAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_MEN/global_finding.json"

OUTPUT_DIR = "late_fusion_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======== 读取 split ========
with open(SPLIT_PATH) as f:
    split = json.load(f)

train_cases = split["train"]
val_cases = split["val"]
test_cases = split["test"]


# ======== 读取所有 modal & global ========
with open(GLI_MODAL) as f:
    gli_modal = json.load(f)

with open(GLI_GLOBAL) as f:
    gli_global = json.load(f)

with open(MEN_MODAL) as f:
    men_modal = json.load(f)

with open(MEN_GLOBAL) as f:
    men_global = json.load(f)


# ======== 合并成一个 dict ========
modal_dict = {}
modal_dict.update(gli_modal)
modal_dict.update(men_modal)

global_dict = {}
global_dict.update(gli_global)
global_dict.update(men_global)


# ======== 构造 structured prompt ========
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


# ======== 构造 dataset ========
def build_dataset(case_list):
    dataset = []

    for case_id in case_list:
        if case_id not in global_dict:
            continue

        # 🔥 单独取每个模态
        t1 = modal_dict.get(case_id + "-t1n", "")
        t2 = modal_dict.get(case_id + "-t2w", "")
        flair = modal_dict.get(case_id + "-t2f", "")
        t1ce = modal_dict.get(case_id + "-t1c", "")

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

    return dataset


train_data = build_dataset(train_cases)
val_data = build_dataset(val_cases)
test_data = build_dataset(test_cases)


# ======== 保存 ========
with open(os.path.join(OUTPUT_DIR, "late_fusion_PullLoss_train.json"), "w") as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "late_fusion_PullLoss_val.json"), "w") as f:
    json.dump(val_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "late_fusion_PullLoss_test.json"), "w") as f:
    json.dump(test_data, f, indent=2)

print("Done!")
print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))
