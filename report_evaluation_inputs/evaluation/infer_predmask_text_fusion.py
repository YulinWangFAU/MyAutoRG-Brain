#!/usr/bin/env python3
"""Run one trained text-fusion model on predicted-mask AutoRG reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed


MODELS = {
    "t5-small": {
        "base": "t5-small",
        "checkpoint": "/home/woody/iwi5/iwi5325h/t5_late_fusion_model_20260218_151904/checkpoint-283",
        "output": "t5_small_predmask.json",
        "lora": False,
    },
    "flan-lora": {
        "base": "google/flan-t5-large",
        "checkpoint": "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_20260218_234219/checkpoint-769",
        "output": "flan_t5_large_lora_predmask.json",
        "lora": True,
    },
    "flan-pullloss": {
        "base": "google/flan-t5-large",
        "checkpoint": "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_PullLoss_20260228_164119/checkpoint-603",
        "output": "flan_t5_large_lora_pullloss_predmask.json",
        "lora": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/text_fusion_predmask"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODELS[args.model]
    set_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this inference job")

    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    test_data = json.loads(args.input.read_text(encoding="utf-8"))
    if len(test_data) != 92:
        raise ValueError(f"Expected 92 test cases, got {len(test_data)}")

    print(f"Model: {args.model}; GPU: {torch.cuda.get_device_name(0)}; cases: {len(test_data)}")
    tokenizer = T5Tokenizer.from_pretrained(config["base"])
    if config["lora"]:
        base_model = T5ForConditionalGeneration.from_pretrained(config["base"])
        base_model.config.use_cache = True
        model = PeftModel.from_pretrained(base_model, checkpoint)
    else:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.to("cuda").eval()

    results = []
    for sample in tqdm(test_data):
        inputs = tokenizer(
            sample["input_text"], return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_length=256, num_beams=4, do_sample=False, early_stopping=True
            )[0].cpu().numpy()
        output = np.where(
            (output >= 0) & (output < tokenizer.vocab_size), output, tokenizer.pad_token_id
        )
        results.append({
            "case_id": sample["case_id"],
            "input": sample["input_text"],
            "prediction": tokenizer.decode(output, skip_special_tokens=True),
            "reference": sample["target_text"],
            "input_source": "predicted_abnormal_mask",
            "model": args.model,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / config["output"]
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(results)} predictions: {output_path}")


if __name__ == "__main__":
    main()
