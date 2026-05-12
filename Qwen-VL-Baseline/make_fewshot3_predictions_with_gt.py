# -*- coding: utf-8 -*-
"""
Created on 2026/5/12 16:34

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Created on 2026/5/10 16:59

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
import json
import os

QWEN_PATH = "/home/woody/iwi5/iwi5325h/qwen25vl_outputs/qwen25vl_fewshot3_fix1_predictions_test.json"

GT_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/evaluation/late_fusion_test_autorg_modal.json"

OUT_PATH = "/home/woody/iwi5/iwi5325h/qwen25vl_outputs/evaluation/qwen25vl_fewshot3_predictions_with_gt.json"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

with open(QWEN_PATH, "r", encoding="utf-8") as f:
    qwen_data = json.load(f)

with open(GT_PATH, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

gt_map = {
    x["case_id"]: x["target_text"]
    for x in gt_data
    if "case_id" in x and "target_text" in x
}

merged = []
missing_gt = []
failed = []

for x in qwen_data:
    case_id = x.get("case_id")
    status = x.get("status", "")

    if status != "success":
        failed.append(case_id)
        continue

    if case_id not in gt_map:
        missing_gt.append(case_id)
        continue

    merged.append({
        "case_id": case_id,
        "prediction": x.get("prediction", ""),
        "reference": gt_map[case_id],
    })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

print("Qwen total:", len(qwen_data))
print("GT total:", len(gt_data))
print("Merged:", len(merged))
print("Failed skipped:", len(failed))
print("Missing GT:", len(missing_gt))
print("Saved to:", OUT_PATH)

if missing_gt[:10]:
    print("Example missing GT:", missing_gt[:10])