# -*- coding: utf-8 -*-
"""
Created on 2026/3/9 16:20

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import json

# ======================
# Paths
# ======================

SPLIT_PATH = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json"

GLI_MODAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_GLI/modal_wise_finding.json"
MEN_MODAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_MEN/modal_wise_finding.json"

OUTPUT_PATH = "evaluation_dataset.json"


# ======================
# Load
# ======================

split = json.load(open(SPLIT_PATH))
gli_gt = json.load(open(GLI_MODAL))
men_gt = json.load(open(MEN_MODAL))

gt_all = {**gli_gt, **men_gt}


# ======================
# case -> split
# ======================

case_split = {}

for s in ["train", "val", "test"]:
    for case in split[s]:
        case_split[case] = s


# ======================
# Build dataset
# ======================

dataset = []

for key, report in gt_all.items():

    parts = key.split("-")

    case_id = "-".join(parts[:4])   # 修正
    modal = parts[4]

    dataset_name = parts[1]

    split_name = case_split.get(case_id)

    if split_name is None:
        continue

    dataset.append({
        "split": split_name,
        "dataset": dataset_name,
        "case_id": case_id,
        "modal": modal,
        "key": key,
        "gt": report
    })


# ======================
# Save
# ======================

with open(OUTPUT_PATH, "w") as f:
    json.dump(dataset, f, indent=2)

print("Saved:", OUTPUT_PATH)
print("Total samples:", len(dataset))