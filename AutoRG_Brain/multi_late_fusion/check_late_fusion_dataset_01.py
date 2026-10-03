# -*- coding: utf-8 -*-
"""
Created on 2026/2/18 11:19
检查训练集代码
@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import json

DATA_PATH = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/late_fusion_data/late_fusion_train.json"
with open(DATA_PATH) as f:
    data = json.load(f)

print("========== Basic Info ==========")
print("Total samples:", len(data))
print()

# 1️⃣ 检查是否 322
if len(data) != 322:
    print("⚠ WARNING: Train size is not 322!")
else:
    print("✔ Train size is correct (322)")
print()

# 2️⃣ 检查是否存在空模态
print("========== Checking Missing Modalities ==========")

missing_cases = []
for sample in data:
    text = sample["input_text"]

    sections = text.split("[")
    for sec in sections:
        if "Imaging]" in sec:
            content = sec.split("]\n")[-1].strip()
            if content == "":
                print("⚠ Empty modality in:", sample["case_id"])

for sample in data:
    text = sample["input_text"]

    # 简单判断是否有空段落
    if "[T1-weighted Imaging]\n\n" in text or \
       "[T2-weighted Imaging]\n\n" in text or \
       "[FLAIR Imaging]\n\n" in text or \
       "[T1CE Imaging]\n\n" in text:
        missing_cases.append(sample["case_id"])

if len(missing_cases) == 0:
    print("✔ No missing modalities detected")
else:
    print("⚠ Missing modalities in:")
    for case in missing_cases[:10]:
        print(case)
print()

# 3️⃣ 检查 input 长度
print("========== Input Length Statistics ==========")

lengths = [len(sample["input_text"]) for sample in data]

print("Min length:", min(lengths))
print("Max length:", max(lengths))
print("Average length:", sum(lengths) / len(lengths))
print()

# 如果超过 2048 字符可能太长
if max(lengths) > 2000:
    print("⚠ WARNING: Some inputs are very long (>2000 characters)")
else:
    print("✔ Input length looks safe")
print()

# 4️⃣ 随机打印一个样本
print("========== Example Sample ==========")
example = data[0]
print("Case ID:", example["case_id"])
print()
print("Input preview:")
print(example["input_text"][:])
print()
print("Target preview:")
print(example["target_text"][:])
print()
