#!/usr/bin/env python3
"""Build a unified 30-configuration text-fusion evaluation table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEEDS = (42, 12345, 2026, 3407, 5901)
MODELS = {
    "t5_small": "T5-Small",
    "flan_t5_large_lora": "Flan-T5-Large-LoRA",
    "flan_t5_large_lora_pullloss": "Flan-T5-Large-LoRA-PullLoss",
}
INPUT_SOURCES = ("gt_abnormal_mask", "predicted_abnormal_mask")
OUTPUT_FIELDS = (
    "category",
    "model",
    "method",
    "seed",
    "input_source",
    "split",
    "case_id",
    "prediction",
    "reference",
    "source_file",
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/home/woody/iwi5/iwi5325h/text_fusion_predictions_multiseed"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "prepared" / "text_fusion_multiseed_reports.csv",
    )
    return parser.parse_args()


def normalize(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{source}: {field} must be a string")
    text = " ".join(value.split())
    if not text:
        raise ValueError(f"{source}: empty {field}")
    return text


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    expected_cases: set[str] | None = None
    expected_references: dict[str, str] = {}

    for input_source in INPUT_SOURCES:
        for method, display_name in MODELS.items():
            for seed in SEEDS:
                path = args.input_root / input_source / f"{method}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, list) or len(payload) != 92:
                    size = len(payload) if isinstance(payload, list) else type(payload).__name__
                    raise ValueError(f"{path}: expected 92 rows, got {size}")

                case_ids: list[str] = []
                for record in payload:
                    case_id = normalize(record.get("case_id"), "case_id", path)
                    prediction = normalize(record.get("prediction"), "prediction", path)
                    reference = normalize(record.get("reference"), "reference", path)
                    if record.get("method") != method:
                        raise ValueError(f"{path}: method mismatch for {case_id}")
                    if int(record.get("seed", -1)) != seed:
                        raise ValueError(f"{path}: seed mismatch for {case_id}")
                    if record.get("input_source") != input_source:
                        raise ValueError(f"{path}: input_source mismatch for {case_id}")
                    if record.get("model") != display_name:
                        raise ValueError(f"{path}: model name mismatch for {case_id}")

                    previous_reference = expected_references.setdefault(case_id, reference)
                    if previous_reference != reference:
                        raise ValueError(f"Reference mismatch across runs for {case_id}")
                    case_ids.append(case_id)
                    rows.append(
                        {
                            "category": "text_level",
                            "model": display_name,
                            "method": method,
                            "seed": seed,
                            "input_source": input_source,
                            "split": "test",
                            "case_id": case_id,
                            "prediction": prediction,
                            "reference": reference,
                            "source_file": str(path),
                        }
                    )

                if len(set(case_ids)) != 92:
                    raise ValueError(f"{path}: duplicate case IDs")
                current_cases = set(case_ids)
                if expected_cases is None:
                    expected_cases = current_cases
                elif current_cases != expected_cases:
                    raise ValueError(f"{path}: case IDs do not align with other runs")

    expected_rows = len(INPUT_SOURCES) * len(MODELS) * len(SEEDS) * 92
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, produced {len(rows)}")
    rows.sort(
        key=lambda row: (
            str(row["input_source"]),
            str(row["model"]),
            int(row["seed"]),
            str(row["case_id"]),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        "Validated 3 models x 5 seeds x 2 input settings x 92 aligned cases"
    )
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote {len(rows)} rows to {jsonl_path}")


if __name__ == "__main__":
    main()
