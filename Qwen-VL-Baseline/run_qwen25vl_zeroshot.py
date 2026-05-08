# -*- coding: utf-8 -*-
"""
Created on 2026/5/8 18:29

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

image_path = "/home/woody/iwi5/iwi5325h/qwen_mri_montage/train/BraTS-GLI-00598-000.png"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name)

image = Image.open(image_path).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "You are an expert neuroradiologist. "
                    "The image is a 2x2 panel of brain MRI sequences from the same patient: "
                    "T1, T1ce, T2, and FLAIR. "
                    "Generate a concise brain MRI radiology report. "
                    "Focus on visible tumor-related findings, enhancement, edema, mass effect, and location. "
                    "Do not invent findings that are not visible."
                ),
            },
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

print("\n===== Qwen2.5-VL Zero-shot Report =====\n")
print(output_text[0])