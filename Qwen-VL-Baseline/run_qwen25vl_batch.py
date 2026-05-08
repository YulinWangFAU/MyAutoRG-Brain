# -*- coding: utf-8 -*-
"""
Created on 2026/5/8 19:20

@author: Yulin Wang
@email: yulin.wang@fau.de

Batch zero-shot inference with Qwen2.5-VL on BraTS MRI montage images.

Output:
- qwen25vl_predictions_train.json
- qwen25vl_predictions_val.json
- qwen25vl_predictions_test.json
"""

import os
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MONTAGE_ROOT = "/home/woody/iwi5/iwi5325h/qwen_mri_montage"
METADATA_CSV = os.path.join(MONTAGE_ROOT, "brats_montage_metadata.csv")

OUTPUT_DIR = "/home/woody/iwi5/iwi5325h/qwen25vl_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


PROMPT = (
    "You are an expert neuroradiologist. "
    "This image is a 2x2 montage of one axial brain MRI slice from the same patient. "
    "Panel layout: top-left = T1, top-right = T1ce, bottom-left = T2, bottom-right = FLAIR. "
    "This case is from a brain tumor MRI dataset. "
    "Generate a concise radiology-style findings section based only on visible image findings. "
    "Describe lesion signal intensity, contrast enhancement, edema, necrosis, mass effect, and lesion location if visible. "
    "If no lesion is clearly visible in this slice, state that no obvious lesion is visible on this selected slice. "
    "Do not include patient name, date, markdown headings, or placeholders. "
    "Do not mention diffusion restriction, hemorrhage, perfusion, spectroscopy, or any sequence that is not provided. "
    "Do not claim the whole brain is normal based on one slice. "
    "Do not make a definitive diagnosis. "
    "Do not invent findings that are not visible."
)


def generate_report(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


def get_image_path(row):
    """
    Directly use image path saved in brats_montage_metadata.csv.
    """
    return row["output_png_path"]


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Reading metadata:", METADATA_CSV)
    df = pd.read_csv(METADATA_CSV)

    print("Metadata columns:", list(df.columns))

    required_cols = {"split", "case_id", "output_png_path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in metadata CSV: {missing_cols}")

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split].copy()

        print(f"\n===== Running split: {split} | n = {len(split_df)} =====")

        results = []

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            case_id = row["case_id"]
            image_path = get_image_path(row)

            if not os.path.exists(image_path):
                print(f"[WARNING] Missing image: {image_path}")
                prediction = ""
                status = "missing_image"
            else:
                try:
                    prediction = generate_report(image_path, model, processor)
                    status = "success"
                except Exception as e:
                    print(f"[ERROR] {case_id}: {e}")
                    prediction = ""
                    status = "error"

            item = {
                "case_id": case_id,
                "split": split,
                "image_path": image_path,
                "largest_tumor_slice_z": int(row["largest_tumor_slice_z"]),
                "max_tumor_area_pixels": int(row["max_tumor_area_pixels"]),
                "prediction": prediction,
                "status": status,
            }

            results.append(item)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"qwen25vl_predictions_{split}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()