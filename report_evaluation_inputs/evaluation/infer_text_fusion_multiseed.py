#!/usr/bin/env python3
"""Run one multi-seed text-fusion model on one held-out test setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed


SEEDS = (42, 12345, 2026, 3407, 5901)
MODELS = {
    "t5_small": {
        "display_name": "T5-Small",
        "base": "t5-small",
        "lora": False,
    },
    "flan_t5_large_lora": {
        "display_name": "Flan-T5-Large-LoRA",
        "base": "google/flan-t5-large",
        "lora": True,
    },
    "flan_t5_large_lora_pullloss": {
        "display_name": "Flan-T5-Large-LoRA-PullLoss",
        "base": "google/flan-t5-large",
        "lora": True,
    },
}
INPUTS = {
    "gt_abnormal_mask": Path(
        "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/"
        "multi_late_fusion/evaluation/late_fusion_test_autorg_modal.json"
    ),
    "predicted_abnormal_mask": Path(
        "/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/"
        "report_evaluation_inputs/evaluation/prepared/late_fusion_test_predmask.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--input-source", choices=INPUTS, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/text_fusion_runs_multiseed"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/woody/iwi5/iwi5325h/text_fusion_predictions_multiseed"
        ),
    )
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-output-length", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any, field: str, case_id: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{case_id}: {field} must be a string")
    text = " ".join(value.split())
    if not text:
        raise ValueError(f"{case_id}: empty {field}")
    return text


def load_test_data(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 92:
        size = len(payload) if isinstance(payload, list) else type(payload).__name__
        raise ValueError(f"Expected a list of 92 test cases in {path}, got {size}")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in payload:
        case_id = normalize_text(record.get("case_id"), "case_id", "unknown")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id in {path}: {case_id}")
        seen.add(case_id)
        rows.append(
            {
                "case_id": case_id,
                "input_text": normalize_text(record.get("input_text"), "input_text", case_id),
                "target_text": normalize_text(
                    record.get("target_text"), "target_text", case_id
                ),
            }
        )
    return rows


def validate_run(run_dir: Path, seed: int, uses_lora: bool) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    config_path = run_dir / "run_config.json"
    result_path = run_dir / "best_result.txt"
    if not config_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(
            f"Incomplete trained run: expected {config_path} and {result_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("seed", -1)) != seed:
        raise ValueError(
            f"Seed mismatch for {run_dir}: requested {seed}, config has {config.get('seed')}"
        )
    if uses_lora:
        required = (run_dir / "adapter_config.json",)
        weight_candidates = (
            run_dir / "adapter_model.safetensors",
            run_dir / "adapter_model.bin",
        )
    else:
        required = (run_dir / "config.json",)
        weight_candidates = (
            run_dir / "model.safetensors",
            run_dir / "pytorch_model.bin",
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not any(path.is_file() for path in weight_candidates):
        expected_weights = " or ".join(str(path) for path in weight_candidates)
        raise FileNotFoundError(
            f"Incomplete model export in {run_dir}; missing={missing}; "
            f"expected weights: {expected_weights}"
        )


def main() -> None:
    args = parse_args()
    config = MODELS[args.model]
    input_path = args.input or INPUTS[args.input_source]
    run_dir = args.runs_root / f"{args.model}_seed{args.seed}"
    output_dir = args.output_root / args.input_source
    output_path = output_dir / f"{args.model}_seed{args.seed}.json"

    validate_run(run_dir, args.seed, bool(config["lora"]))
    test_data = load_test_data(input_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing predictions: {output_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for text-fusion inference")

    set_seed(args.seed)
    print(
        f"model={args.model} seed={args.seed} input_source={args.input_source} "
        f"gpu={torch.cuda.get_device_name(0)} cases={len(test_data)}"
    )
    print(f"run_dir={run_dir}")
    print(f"input={input_path}")
    print(f"output={output_path}")

    tokenizer = T5Tokenizer.from_pretrained(config["base"])
    if config["lora"]:
        base_model = T5ForConditionalGeneration.from_pretrained(config["base"])
        base_model.config.use_cache = True
        model = PeftModel.from_pretrained(base_model, run_dir)
    else:
        model = T5ForConditionalGeneration.from_pretrained(run_dir)
    model.to("cuda").eval()

    results: list[dict[str, object]] = []
    for sample in tqdm(test_data):
        inputs = tokenizer(
            sample["input_text"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to("cuda")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_length=args.max_output_length,
                num_beams=args.num_beams,
                do_sample=False,
                early_stopping=True,
            )[0].cpu().numpy()
        output_ids = np.where(
            (output_ids >= 0) & (output_ids < tokenizer.vocab_size),
            output_ids,
            tokenizer.pad_token_id,
        )
        prediction = " ".join(
            tokenizer.decode(output_ids, skip_special_tokens=True).split()
        )
        if not prediction:
            raise ValueError(f"Generated an empty report for {sample['case_id']}")
        results.append(
            {
                "category": "text_level",
                "model": config["display_name"],
                "method": args.model,
                "seed": args.seed,
                "input_source": args.input_source,
                "split": "test",
                "case_id": sample["case_id"],
                "input": sample["input_text"],
                "prediction": prediction,
                "reference": sample["target_text"],
                "model_dir": str(run_dir),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(output_path)
    print(f"Saved {len(results)} predictions: {output_path}")


if __name__ == "__main__":
    main()
