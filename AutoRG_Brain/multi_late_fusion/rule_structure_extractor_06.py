# -*- coding: utf-8 -*-
"""
Created on 2026/2/19 23:07
从现在开始就是结构化输入，
step1:抽结构
step2:合并结构
step3:结构->文本
但是写代码提取Schema失败，因为有各种表达意思的语言模式
用这个不靠谱
@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# -*- coding: utf-8 -*-
"""
Rule-based MRI Structure Extractor
Author: Yulin Wang
Description:
    Extract lesion-level structured schema from modality-wise MRI findings.
"""

import json
import re
import os


# =========================
# 1. 初始化 Schema
# =========================

def init_schema():
    return {
        "lesion_id": "lesion_1",

        "core": {
            "location": {
                "lobe": None,
                "side": None,
                "deep_structures": []
            },
            "size_mm": None,
            "boundary": None
        },

        "signal": {
            "T1": None,
            "T2": None,
            "FLAIR": None,
            "T1C_enhancement": {
                "pattern": None,
                "degree": None
            }
        },

        "associated_findings": {
            "edema": {
                "present": False,
                "extent": None
            }
        },

        "mass_effect": {
            "midline_shift": {
                "present": False,
                "direction": None
            },
            "ventricle_compression": {
                "left_lateral": False,
                "right_lateral": False,
                "anterior_horn": False,
                "posterior_horn": False
            }
        }
    }


# =========================
# 2. 文本分块
# =========================

def split_modalities(text):
    pattern = r"\[(.*?)\]\n(.*?)(?=\n\[|$)"
    matches = re.findall(pattern, text, re.DOTALL)

    blocks = {}
    for modality, content in matches:
        blocks[modality.strip()] = content.strip()

    return blocks


# =========================
# 3. 抽取函数
# =========================

def extract_location(text, schema):
    lobes = ["frontal", "temporal", "parietal", "occipital"]
    sides = ["left", "right", "bilateral"]

    lower_text = text.lower()

    for side in sides:
        if side in lower_text:
            schema["core"]["location"]["side"] = side

    for lobe in lobes:
        if lobe in lower_text:
            schema["core"]["location"]["lobe"] = lobe + " lobe"


def extract_size(text, schema):
    match = re.search(r'(\d+)\s*[x\*]\s*(\d+)\s*[x\*]\s*(\d+)\s*mm', text)
    if match:
        schema["core"]["size_mm"] = [
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3))
        ]


def extract_boundary(text, schema):
    lower_text = text.lower()

    if "clear boundary" in lower_text:
        schema["core"]["boundary"] = "clear"
    elif "unclear" in lower_text or "indistinct" in lower_text:
        schema["core"]["boundary"] = "unclear"


def extract_signal(modality, text, schema):

    lower_text = text.lower()

    value = None
    if "slightly high" in lower_text:
        value = "slightly high"
    elif "slightly low" in lower_text:
        value = "slightly low"
    elif "high signal" in lower_text:
        value = "high"
    elif "low signal" in lower_text:
        value = "low"

    if "T1-weighted" in modality:
        schema["signal"]["T1"] = value
    elif "T2-weighted" in modality:
        schema["signal"]["T2"] = value
    elif "FLAIR" in modality:
        schema["signal"]["FLAIR"] = value


def extract_enhancement(modality, text, schema):

    if "T1C" not in modality and "contrast" not in modality:
        return

    lower_text = text.lower()

    if "ring-like" in lower_text:
        schema["signal"]["T1C_enhancement"]["pattern"] = "ring-like"
    elif "uneven" in lower_text:
        schema["signal"]["T1C_enhancement"]["pattern"] = "uneven"
    elif "nodular" in lower_text:
        schema["signal"]["T1C_enhancement"]["pattern"] = "nodular"

    if "mild" in lower_text:
        schema["signal"]["T1C_enhancement"]["degree"] = "mild"
    elif "marked" in lower_text:
        schema["signal"]["T1C_enhancement"]["degree"] = "marked"
    elif "moderate" in lower_text:
        schema["signal"]["T1C_enhancement"]["degree"] = "moderate"


def extract_edema(text, schema):
    lower_text = text.lower()

    if "edema" in lower_text:
        schema["associated_findings"]["edema"]["present"] = True

        if "extensive" in lower_text:
            schema["associated_findings"]["edema"]["extent"] = "extensive"
        elif "mild" in lower_text:
            schema["associated_findings"]["edema"]["extent"] = "mild"
        elif "moderate" in lower_text:
            schema["associated_findings"]["edema"]["extent"] = "moderate"


def extract_midline_shift(text, schema):
    lower_text = text.lower()

    if "midline" in lower_text:
        schema["mass_effect"]["midline_shift"]["present"] = True

        if "left" in lower_text:
            schema["mass_effect"]["midline_shift"]["direction"] = "left"
        elif "right" in lower_text:
            schema["mass_effect"]["midline_shift"]["direction"] = "right"


def extract_ventricle(text, schema):
    lower_text = text.lower()

    if "lateral ventricle" in lower_text:
        if "left" in lower_text:
            schema["mass_effect"]["ventricle_compression"]["left_lateral"] = True
        if "right" in lower_text:
            schema["mass_effect"]["ventricle_compression"]["right_lateral"] = True

    if "anterior horn" in lower_text:
        schema["mass_effect"]["ventricle_compression"]["anterior_horn"] = True

    if "posterior horn" in lower_text:
        schema["mass_effect"]["ventricle_compression"]["posterior_horn"] = True


# =========================
# 4. 主抽取流程
# =========================

def extract_structure(full_text):

    schema = init_schema()
    blocks = split_modalities(full_text)

    for modality, content in blocks.items():

        extract_location(content, schema)
        extract_size(content, schema)
        extract_boundary(content, schema)
        extract_signal(modality, content, schema)
        extract_enhancement(modality, content, schema)
        extract_edema(content, schema)
        extract_midline_shift(content, schema)
        extract_ventricle(content, schema)

    return schema


# =========================
# 5. 主程序入口
# =========================

def main():

    input_path = "/Users/wangyulin/LLM-MRI-THESIS-PROJECT/RadGenome-Brain_MRI/BraTS_GLI/modal_wise_finding.json"
    output_path = "late_fusion_data/structured_output.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    # 1️⃣ 按 case 聚合
    case_dict = {}

    for full_id, content in data.items():
        case_id = "-".join(full_id.split("-")[:-1])  # 去掉 t1n t2w 等

        if case_id not in case_dict:
            case_dict[case_id] = []

        case_dict[case_id].append((full_id, content))

    structured_results = {}

    # 2️⃣ 对每个 case 抽取结构
    for case_id, modalities in case_dict.items():

        schema = init_schema()

        for full_id, content in modalities:

            # 根据 full_id 判断 modality
            if "t1n" in full_id.lower():
                modality_name = "T1-weighted Imaging"
            elif "t2w" in full_id.lower():
                modality_name = "T2-weighted Imaging"
            elif "t2f" in full_id.lower():
                modality_name = "FLAIR Imaging"
            elif "t1c" in full_id.lower():
                modality_name = "T1C Imaging"
            else:
                modality_name = "Unknown"

            extract_location(content, schema)
            extract_size(content, schema)
            extract_boundary(content, schema)
            extract_signal(modality_name, content, schema)
            extract_enhancement(modality_name, content, schema)
            extract_edema(content, schema)
            extract_midline_shift(content, schema)
            extract_ventricle(content, schema)

        structured_results[case_id] = schema

    with open(output_path, "w") as f:
        json.dump(structured_results, f, indent=4)

    print("Structure extraction completed.")


if __name__ == "__main__":
    main()
