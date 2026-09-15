#!/usr/bin/env python3
"""Convert predicted-mask text-fusion outputs to the unified RaTEScore schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = (
    "category",
    "model",
    "input_source",
    "split",
    "case_id",
    "prediction",
    "reference",
    "source_file",
)
SOURCES = {
    "t5_small_predmask.json": "T5-Small",
    "flan_t5_large_lora_predmask.json": "Flan-T5-Large-LoRA",
    "flan_t5_large_lora_pullloss_predmask.json": "Flan-T5-Large-LoRA-PullLoss",
}


def normalize(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected report text, got {type(value).__name__}")
    text = " ".join(value.split())
    if not text:
        raise ValueError("Encountered an empty report")
    return text


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/text_fusion_predmask"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "prepared" / "predmask_text_fusion_reports.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    expected_cases: set[str] | None = None

    for filename, model_name in SOURCES.items():
        path = args.input_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload) != 92:
            raise ValueError(f"{filename}: expected 92 rows, got {len(payload)}")
        case_ids = [record["case_id"] for record in payload]
        if len(set(case_ids)) != 92:
            raise ValueError(f"{filename}: duplicate case IDs")
        current_cases = set(case_ids)
        if expected_cases is None:
            expected_cases = current_cases
        elif current_cases != expected_cases:
            raise ValueError(f"{filename}: case IDs do not align with other models")

        for record in payload:
            rows.append({
                "category": "text_level",
                "model": model_name,
                "input_source": "predicted_abnormal_mask",
                "split": "test",
                "case_id": record["case_id"],
                "prediction": normalize(record["prediction"]),
                "reference": normalize(record["reference"]),
                "source_file": str(path),
            })

    rows.sort(key=lambda row: (row["model"], row["case_id"]))
    if len(rows) != 276:
        raise ValueError(f"Expected 276 rows, produced {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with args.output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Validated 3 text-fusion models and 92 aligned cases per model")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
