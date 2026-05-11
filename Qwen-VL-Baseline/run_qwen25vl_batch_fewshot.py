# -*- coding: utf-8 -*-
"""
Batch few-shot inference with Qwen2.5-VL on BraTS MRI montage images.

Few-shot examples:
1. GLI severe case
2. GLI moderate / no-shift case
3. MEN typical small enhancing nodule

Output:
- qwen25vl_fewshot3_fix_predictions_train.json
- qwen25vl_fewshot3_fix_predictions_val.json
- qwen25vl_fewshot3_fix_predictions_test.json
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


FEW_SHOT_CASES = [
    {
        "case_id": "BraTS-GLI-00158-000",
        "target_text": (
            "On the right frontal-parietal-temporal lobe, an irregular abnormal signal focus is visible, "
            "presenting with low signal on T1W with sparse high signal, mixed high and low signal on T2W "
            "and FLAIR sequences, with significant ring-like enhancement on T1C. The border is not clear, "
            "with dimensions approximately 77*104*71mm. Extensive cerebral edema is observed in the surrounding "
            "brain parenchyma, with compression of the right lateral ventricle and deviation of midline structures "
            "to the left."
        )
    },
    {
        "case_id": "BraTS-GLI-00024-000",
        "target_text": (
            "An abnormal signal focus can be seen in the left temporal lobe, showing slightly low signal on "
            "T1-weighted images, high signal on T2-weighted images, and high signal on FLAIR images, with some "
            "mixed signals inside, ring-enhancement on post-contrast T1 images, and unclear boundaries. The size "
            "is approximately 53*81*65 mm. There is surrounding brain tissue edema; no midline shift is present."
        )
    },
    {
        "case_id": "BraTS-MEN-01052-000",
        "target_text": (
            "A small nodular abnormal signal focus is seen under the right frontal bone plate, showing isointense "
            "signal on T1-weighted and T2-weighted images, slightly high signal on FLAIR images, and obvious "
            "enhancement on T1-weighted images after contrast. The lesion is well-defined and measures approximately "
            "9*14*16 mm. No edema or midline shift is observed."
        )
    },
]


def get_image_path(row):
    return row["output_png_path"]


def build_few_shot_examples(df):
    case_to_image_path = {
        row["case_id"]: row["output_png_path"]
        for _, row in df.iterrows()
    }

    few_shot_examples = []

    for ex in FEW_SHOT_CASES:
        case_id = ex["case_id"]

        if case_id not in case_to_image_path:
            raise ValueError(f"Few-shot case not found in metadata CSV: {case_id}")

        image_path = case_to_image_path[case_id]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Few-shot image not found: {image_path}")

        few_shot_examples.append({
            "case_id": case_id,
            "image_path": image_path,
            "target_text": ex["target_text"],
        })

    return few_shot_examples


def generate_report(image_path, model, processor, few_shot_examples):
    query_image = Image.open(image_path).convert("RGB")

    messages = []
    all_images = []

    # 1. Add few-shot examples
    for ex in few_shot_examples:
        ex_image = Image.open(ex["image_path"]).convert("RGB")

        # User gives example image + prompt
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        })

        # Assistant gives example target report
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": ex["target_text"]},
            ],
        })

        all_images.append(ex_image)

    # 2. Add current query image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    })

    all_images.append(query_image)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=all_images,
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

    required_cols = {
        "split",
        "case_id",
        "output_png_path",
        "largest_tumor_slice_z",
        "max_tumor_area_pixels",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in metadata CSV: {missing_cols}")

    few_shot_examples = build_few_shot_examples(df)

    print("\nFew-shot examples:")
    for ex in few_shot_examples:
        print(ex["case_id"], "->", ex["image_path"])

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
                    prediction = generate_report(
                        image_path=image_path,
                        model=model,
                        processor=processor,
                        few_shot_examples=few_shot_examples
                    )
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
                "few_shot_case_ids": [ex["case_id"] for ex in few_shot_examples],
            }

            results.append(item)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"qwen25vl_fewshot3_fix_predictions_{split}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()