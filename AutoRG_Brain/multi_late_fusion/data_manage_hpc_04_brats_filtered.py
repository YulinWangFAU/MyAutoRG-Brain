# -*- coding: utf-8 -*-
"""
Created on 2026/2/18 18:17

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import json
import shutil

# ===== 路径 =====
json_path = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json"

gli_root = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/BraTS 2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
men_root = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/BraTS 2023/BraTS-MEN-Train"

output_root = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/BraTS_filtered"

# ===== 读取json =====
with open(json_path, 'r') as f:
    split_dict = json.load(f)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, split), exist_ok=True)

    for subject in split_dict[split]:

        if "GLI" in subject:
            src = os.path.join(gli_root, subject)
        else:
            src = os.path.join(men_root, subject)

        dst = os.path.join(output_root, split, subject)

        if os.path.exists(src):
            shutil.copytree(src, dst)
        else:
            print("Missing:", subject)
