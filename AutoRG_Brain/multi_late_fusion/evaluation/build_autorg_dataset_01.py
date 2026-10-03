# -*- coding: utf-8 -*-
"""
Created on 2026/3/13 16:19

@author: Yulin Wang
@email: yulin.wang@fau.de
读取 AutoRG-Brain 生成的四个单模态报告，
把同一个病例的四个模态报告组合成一个文本提示词，
并将人工标注的全局报告作为目标答案，
最终生成一个用于测试 Late Fusion 文本模型的 JSON 数据集。
{
  "case_id": "BraTS-GLI-00006-000",
  "input_text": "四个 AutoRG 单模态报告组成的 prompt",
  "target_text": "该病例的人工 GT 全局报告"
}
AUTORG_PATH = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/Autorg_output/autorg_predictions/autorg_predictions_test.json"
"""

import json
from collections import defaultdict

# =========================
# 路径
# =========================

AUTORG_PATH = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/Autorg_output/autorg_predictions/autorg_predictions_test.json"


GLI_GLOBAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_GLI/global_finding.json"
MEN_GLOBAL = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_MEN/global_finding.json"

OUTPUT_PATH = "late_fusion_test_autorg_modal.json"


# =========================
# 读取 AutoRG modal predictions
# =========================

with open(AUTORG_PATH) as f:
    autorg_modal = json.load(f)


# =========================
# 读取 global reports
# =========================

with open(GLI_GLOBAL) as f:
    gli_global = json.load(f)

with open(MEN_GLOBAL) as f:
    men_global = json.load(f)

# 合并
global_dict = {}
global_dict.update(gli_global)
global_dict.update(men_global)


print("Total global reports:", len(global_dict))


# =========================
# 按 case 聚合 modal reports
# =========================

case_dict = defaultdict(dict)

for full_id, report in autorg_modal.items():

    case = full_id.rsplit("-",1)[0]
    modality = full_id.rsplit("-",1)[1]

    case_dict[case][modality] = report


print("Total cases with AutoRG modal:", len(case_dict))


# =========================
# 构造 prompt
# =========================

def build_prompt(modals):

    t1 = modals.get("t1n","")
    t2 = modals.get("t2w","")
    flair = modals.get("t2f","")
    t1ce = modals.get("t1c","")

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


# =========================
# 构造 dataset
# =========================

dataset = []

for case_id, modals in case_dict.items():

    # 必须有 global report
    if case_id not in global_dict:
        continue

    # 必须有4个模态
    if len(modals) < 4:
        continue

    dataset.append({
        "case_id": case_id,
        "input_text": build_prompt(modals),
        "target_text": global_dict[case_id]
    })


print("Final dataset size:", len(dataset))


# =========================
# 保存
# =========================

with open(OUTPUT_PATH,"w") as f:
    json.dump(dataset,f,indent=2)

print("Saved:",OUTPUT_PATH)