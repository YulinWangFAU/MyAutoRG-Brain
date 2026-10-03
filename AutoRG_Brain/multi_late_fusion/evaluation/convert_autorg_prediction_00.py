# -*- coding: utf-8 -*-
"""
Created on 2026/3/9 16:55

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import json

# =========================
# 输入路径
# =========================

PRED_PATH = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/Autorg_output/autorg_output_test/pred_report.json"

# 输出路径
OUTPUT_PATH = "autorg_predictions_test.json"


# =========================
# 读取 AutoRG 输出
# =========================

data = json.load(open(PRED_PATH))

pred_dict = {}

for item in data:

    # image 是 list
    image_path = item["image"][0]

    filename = image_path.split("/")[-1]

    key = filename.replace(".nii.gz", "")

    pred_text = item["pred_report"]

    pred_dict[key] = pred_text


# =========================
# 保存
# =========================

with open(OUTPUT_PATH, "w") as f:
    json.dump(pred_dict, f, indent=2)

print("Saved:", OUTPUT_PATH)
print("Total predictions:", len(pred_dict))