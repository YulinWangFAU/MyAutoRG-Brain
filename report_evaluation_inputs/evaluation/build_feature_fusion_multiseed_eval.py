#!/usr/bin/env python3
"""Build the unified evaluation table for multi-seed feature fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SEEDS = (42, 12345, 2026, 3407, 5901)
MODELS = {
    "weighted_mean": "Weighted-Mean",
    "concat_projection": "Concat-Projection",
    "token_gated": "Token-Gated",
}
INPUT_SOURCES = ("gt_abnormal_mask", "predicted_abnormal_mask")
FIELDS = (
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
            "/home/woody/iwi5/iwi5325h/feature_fusion_predictions_multiseed"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "prepared" / "feature_fusion_multiseed_reports.csv",
    )
    return parser.parse_args()


def normalize(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: {field} must be a string")
    text = " ".join(value.split())
    if not text:
        raise ValueError(f"{path}: empty {field}")
    return text


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    expected_cases: set[str] | None = None
    expected_references: dict[str, str] = {}

    for source in INPUT_SOURCES:
        for method, model_name in MODELS.items():
            for seed in SEEDS:
                path = args.input_root / source / f"{method}_seed{seed}" / "pred_test.json"
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("fusion_type") != method:
                    raise ValueError(f"{path}: fusion_type mismatch")
                if payload.get("split") != "test":
                    raise ValueError(f"{path}: expected test split")
                results = payload.get("results")
                if not isinstance(results, list) or len(results) != 92:
                    size = len(results) if isinstance(results, list) else type(results).__name__
                    raise ValueError(f"{path}: expected 92 results, got {size}")

                case_ids: list[str] = []
                for result in results:
                    case_id = normalize(result.get("case_id"), "case_id", path)
                    prediction = normalize(result.get("pred_report"), "pred_report", path)
                    reference = normalize(
                        result.get("reference_report"), "reference_report", path
                    )
                    previous_reference = expected_references.setdefault(case_id, reference)
                    if previous_reference != reference:
                        raise ValueError(f"Reference mismatch across runs for {case_id}")
                    case_ids.append(case_id)
                    rows.append(
                        {
                            "category": "feature_level",
                            "model": model_name,
                            "method": method,
                            "seed": seed,
                            "input_source": source,
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
                    raise ValueError(f"{path}: case IDs do not align")

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
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path = args.output.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Validated 3 feature-fusion models x 5 seeds x 2 input settings")
    print("Validated 92 aligned held-out cases per configuration")
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote {len(rows)} rows to {jsonl_path}")


if __name__ == "__main__":
    main()
