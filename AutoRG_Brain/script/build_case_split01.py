# -*- coding: utf-8 -*-
"""
Created on 2026/2/15

Case-level split (GLI + MEN only)
"""

import json

# ===== 1. 路径 =====
input_path = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/train_val_test_split.json"
output_path = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json"

# ===== 2. 读取原始 split =====
with open(input_path, "r") as f:
    data = json.load(f)

new_split = {}
statistics = {}

print("\n" + "="*60)
print("STEP 1: Build GLI + MEN case-level split")
print("="*60)

for split_name in ["train", "val", "test"]:
    samples = data[split_name]

    case_ids = set()
    gli_cases = set()
    men_cases = set()

    gli_modal_count = 0
    men_modal_count = 0

    for sample in samples:
        sample = sample.strip()
        if not sample:
            continue

        # 🔥 只保留 GLI 和 MEN
        if ("GLI" not in sample) and ("MEN" not in sample):
            continue

        case_id = sample.rsplit("-", 1)[0]
        case_ids.add(case_id)

        if "GLI" in sample:
            gli_cases.add(case_id)
            gli_modal_count += 1

        if "MEN" in sample:
            men_cases.add(case_id)
            men_modal_count += 1

    new_split[split_name] = sorted(list(case_ids))

    statistics[split_name] = {
        "modal_wise_count": gli_modal_count + men_modal_count,
        "case_wise_count": len(case_ids),
        "GLI_case_count": len(gli_cases),
        "MEN_case_count": len(men_cases),
        "GLI_modal_count": gli_modal_count,
        "MEN_modal_count": men_modal_count,
    }

# ===== 3. 保存新的 split =====
with open(output_path, "w") as f:
    json.dump(new_split, f, indent=4)

# ===== 4. 打印统计 =====
print("\n" + "="*60)
print("GLI + MEN STATISTICS")
print("="*60)

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")
    print("-"*40)
    print("Modal-wise samples :", statistics[split]["modal_wise_count"])
    print("Case-wise samples  :", statistics[split]["case_wise_count"])
    print("GLI case count     :", statistics[split]["GLI_case_count"])
    print("MEN case count     :", statistics[split]["MEN_case_count"])
    print("GLI modal count    :", statistics[split]["GLI_modal_count"])
    print("MEN modal count    :", statistics[split]["MEN_modal_count"])

print("\nDone building GLI + MEN case-level split.")
