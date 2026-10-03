# -*- coding: utf-8 -*-
"""
Check multi-modal completeness for GLI + MEN
"""

import json
from collections import defaultdict

# ===== 路径 =====
input_path = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/train_val_test_split.json"

# 期望的4个模态
expected_modalities = {"t1c", "t1n", "t2f", "t2w"}

# ===== 读取原始 split =====
with open(input_path, "r") as f:
    data = json.load(f)

print("\n" + "="*60)
print("STEP 1: Build case → modality mapping (GLI + MEN)")
print("="*60)

for split_name in ["train", "val", "test"]:

    samples = data[split_name]

    case_modal_map = defaultdict(set)

    for sample in samples:
        sample = sample.strip()
        if not sample:
            continue

        # 只保留 GLI + MEN
        if ("GLI" not in sample) and ("MEN" not in sample):
            continue

        case_id, modal = sample.rsplit("-", 1)
        case_modal_map[case_id].add(modal.lower())

    print(f"\n{split_name.upper()}")
    print("-"*40)

    total_cases = len(case_modal_map)
    complete_cases = 0
    incomplete_cases = []

    for case_id, modals in case_modal_map.items():

        if modals == expected_modalities:
            complete_cases += 1
        else:
            missing = expected_modalities - modals
            incomplete_cases.append((case_id, modals, missing))

    print("Total cases       :", total_cases)
    print("Complete (4 mods) :", complete_cases)
    print("Incomplete cases  :", len(incomplete_cases))

    if incomplete_cases:
        print("\n⚠ Missing modality details:")
        for case_id, modals, missing in incomplete_cases[:10]:
            print("Case:", case_id)
            print("Have:", sorted(modals))
            print("Miss:", sorted(missing))
            print()

print("\nDone.")

# 保存 mapping 表
mapping_output = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/gli_men_multi_modal_mapping.json"

final_mapping = {}

for split_name in ["train", "val", "test"]:

    samples = data[split_name]
    case_modal_map = defaultdict(list)

    for sample in samples:
        sample = sample.strip()
        if not sample:
            continue

        if ("GLI" not in sample) and ("MEN" not in sample):
            continue

        case_id, modal = sample.rsplit("-", 1)
        case_modal_map[case_id].append(modal.lower())

    final_mapping[split_name] = dict(case_modal_map)

with open(mapping_output, "w") as f:
    json.dump(final_mapping, f, indent=4)

print("Saved multi-modal mapping table.")
