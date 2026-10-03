# -*- coding: utf-8 -*-
"""
Created on 2026/2/20 16:12

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import json
import os
from tqdm import tqdm
import time
from openai import OpenAI

# 如果你用DeepSeek，可以这样配置
# client = OpenAI(
#     api_key="your-api-key",
#     base_url="https://api.deepseek.com"
# )

# 配置路径
SPLIT_PATH = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json"
GLI_MODAL = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/RadGenome-Brain_MRI/BraTS_GLI/modal_wise_finding.json"
MEN_MODAL = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/RadGenome-Brain_MRI/BraTS_MEN/modal_wise_finding.json"

# 输出路径
OUTPUT_DIR = "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/multi_late_fusion/structured_modal_data"
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train_modal_structured.json")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "val_modal_structured.json")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "test_modal_structured.json")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化OpenAI客户端 (用你的实际API key)
client = OpenAI(
    api_key="your-api-key-here",
    base_url="https://api.openai.com/v1"  # 或者用DeepSeek的URL
)

# ==================== PROMPT ====================
SYSTEM_PROMPT = """You are an expert radiologist specializing in brain MRI analysis. Your task is to extract structured information from a single-sentence MRI finding and convert it into a JSON format.

The input will be a sentence describing findings from ONE specific MRI sequence (T1, T2, FLAIR, or T1C).

You must follow these rules strictly:
1. Only extract information that is explicitly mentioned in the input sentence. Do NOT add any information that is not present.
2. For the "signal_characteristics" object: ONLY fill the field corresponding to the input modality. All other signal fields MUST be null.
3. Use the exact enum values provided in the schema.
4. If a piece of information is not mentioned, set its value to null.

Output JSON Schema:
{
  "modality": "Must be one of: T1, T2, FLAIR, T1C",
  "lesion_exists": true or false,
  "location": "Anatomical location (e.g., left frontal lobe). null if not mentioned.",
  "size": {
    "description": "The exact text describing size, e.g., 'approximately 76 x 72 x 69 mm'. null if not mentioned.",
    "dimensions": "If 3D dimensions are given, extract as [x, y, z] array. Otherwise null."
  },
  "signal_characteristics": {
    "t1_signal": "Only if modality is T1. Enum: ['hyperintense', 'hypointense', 'isointense', 'mixed', null]",
    "t2_signal": "Only if modality is T2. Enum: ['hyperintense', 'hypointense', 'isointense', 'mixed', null]",
    "flair_signal": "Only if modality is FLAIR. Enum: ['hyperintense', 'hypointense', 'isointense', 'mixed', null]",
    "t1c_enhancement": "Only if modality is T1C. Describe enhancement pattern as text, e.g., 'marked ring-like enhancement'. null if not T1C."
  },
  "boundary": {
    "clarity": "Enum: ['clear', 'unclear', 'ill-defined', 'poorly defined', null]",
    "description": "The exact text describing boundary. null if not mentioned."
  },
  "peritumoral_status": {
    "edema": "Enum: ['present', 'absent', null]",
    "edema_description": "The exact text describing edema. null if not mentioned."
  },
  "mass_effect": {
    "present": "Enum: ['present', 'absent', null]. 'present' if any mass effect (ventricular compression, midline shift) is mentioned.",
    "details": "Specific description of mass effect, e.g., 'compression of left lateral ventricle'. null if not mentioned or mass_effect is 'absent'."
  },
  "other_features": "Any other notable features like 'fluid level', 'crosses the midline', 'dural tail sign'. null if none."
}

Here are two examples:

Example 1:
Input: "On the T2-weighted sequence, the lesion in the left frontal lobe is manifested as a high signal with unclear borders. There is associated extensive edema in the surrounding brain tissue."
Output:
{
  "modality": "T2",
  "lesion_exists": true,
  "location": "left frontal lobe",
  "size": {
    "description": null,
    "dimensions": null
  },
  "signal_characteristics": {
    "t1_signal": null,
    "t2_signal": "hyperintense",
    "flair_signal": null,
    "t1c_enhancement": null
  },
  "boundary": {
    "clarity": "unclear",
    "description": "unclear borders"
  },
  "peritumoral_status": {
    "edema": "present",
    "edema_description": "extensive edema in the surrounding brain tissue"
  },
  "mass_effect": {
    "present": null,
    "details": null
  },
  "other_features": null
}

Example 2:
Input: "After contrast administration (T1C), the lesion in the left frontal lobe shows marked ring-like enhancement, with compression of the left lateral ventricle and rightward shift of midline structures."
Output:
{
  "modality": "T1C",
  "lesion_exists": true,
  "location": "left frontal lobe",
  "size": {
    "description": null,
    "dimensions": null
  },
  "signal_characteristics": {
    "t1_signal": null,
    "t2_signal": null,
    "flair_signal": null,
    "t1c_enhancement": "marked ring-like enhancement"
  },
  "boundary": {
    "clarity": null,
    "description": null
  },
  "peritumoral_status": {
    "edema": null,
    "edema_description": null
  },
  "mass_effect": {
    "present": "present",
    "details": "compression of the left lateral ventricle, rightward shift of midline structures"
  },
  "other_features": null
}"""


# ==================== 辅助函数 ====================

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_modality_from_key(key):
    """从key中提取模态信息"""
    if key.endswith('-t1c'):
        return 'T1C'
    elif key.endswith('-t1n'):
        return 'T1'
    elif key.endswith('-t2f'):
        return 'FLAIR'
    elif key.endswith('-t2w'):
        return 'T2'
    else:
        return None


def call_llm(text, max_retries=3):
    """调用LLM获取结构化输出"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # 或者 "deepseek-chat"
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Input: {text}\n\nOutput:"}
                ],
                temperature=0.1,  # 低温度确保一致性
                max_tokens=1000
            )

            # 提取JSON
            content = response.choices[0].message.content

            # 尝试解析JSON
            # 有时模型会返回带有markdown格式的JSON，需要清理
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            return json.loads(content)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"Failed after {max_retries} attempts for text: {text[:100]}...")
                return None


def process_split(split_name, case_ids, modal_data, output_file):
    """处理一个分割集（train/val/test）的所有病例"""
    print(f"\nProcessing {split_name} split...")

    results = []

    # 创建进度条
    pbar = tqdm(case_ids, desc=f"{split_name}")

    for case_id in pbar:
        # 更新进度条描述
        pbar.set_description(f"{split_name} - {case_id}")

        # 找到这个case的所有模态key
        modal_keys = [key for key in modal_data.keys() if key.startswith(case_id)]

        # 按模态排序，确保顺序一致
        modal_keys.sort()

        for key in modal_keys:
            text = modal_data[key]
            modality = extract_modality_from_key(key)

            # 调用LLM
            structured = call_llm(text)

            if structured:
                # 添加元数据
                structured['case_id'] = case_id
                structured['modal_key'] = key
                structured['original_text'] = text

                results.append(structured)

                # 每处理10条保存一次，避免丢失
                if len(results) % 10 == 0:
                    save_json(results, output_file + ".tmp")

            # 添加小延迟避免API限流
            time.sleep(0.5)

    # 保存最终结果
    save_json(results, output_file)
    print(f"Saved {len(results)} items to {output_file}")

    return results


# ==================== 主函数 ====================

def main():
    print("Loading data...")

    # 加载分割文件
    split_data = load_json(SPLIT_PATH)
    train_ids = split_data['train']
    val_ids = split_data['val']
    test_ids = split_data['test']

    print(f"Train cases: {len(train_ids)}")
    print(f"Val cases: {len(val_ids)}")
    print(f"Test cases: {len(test_ids)}")

    # 加载模态数据
    gli_modal = load_json(GLI_MODAL)
    men_modal = load_json(MEN_MODAL)

    # 合并所有模态数据
    all_modal_data = {**gli_modal, **men_modal}
    print(f"Total modal entries: {len(all_modal_data)}")

    # 处理训练集
    train_results = process_split("TRAIN", train_ids, all_modal_data, TRAIN_OUTPUT)

    # 处理验证集
    val_results = process_split("VAL", val_ids, all_modal_data, VAL_OUTPUT)

    # 处理测试集
    test_results = process_split("TEST", test_ids, all_modal_data, TEST_OUTPUT)

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Train: {len(train_results)} structured items")
    print(f"Val: {len(val_results)} structured items")
    print(f"Test: {len(test_results)} structured items")
    print(f"Total: {len(train_results) + len(val_results) + len(test_results)} items")
    print("=" * 50)


if __name__ == "__main__":
    main()