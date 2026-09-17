#!/usr/bin/env python3
"""Convert predicted-mask feature-fusion reports to the unified RaTEScore schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
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
    "mean": "Mean",
    "weighted_mean": "Weighted-Mean",
    "concat_projection": "Concat-Projection",
    "token_gated": "Token-Gated",
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
        "--input-root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/feature_fusion_predmask"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "prepared" / "predmask_feature_fusion_reports.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    expected_cases: set[str] | None = None

    for fusion, model_name in SOURCES.items():
        path = args.input_root / fusion / "pred_test.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fusion_type") != fusion:
            raise ValueError(f"{path}: fusion_type does not equal {fusion!r}")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 92:
            raise ValueError(f"{path}: expected 92 results")
        case_ids = [row["case_id"] for row in results]
        if len(set(case_ids)) != 92:
            raise ValueError(f"{path}: duplicate case IDs")
        current_cases = set(case_ids)
        if expected_cases is None:
            expected_cases = current_cases
        elif current_cases != expected_cases:
            raise ValueError(f"{path}: case IDs do not align with other fusion models")

        for result in results:
            rows.append({
                "category": "feature_level",
                "model": model_name,
                "input_source": "predicted_abnormal_mask",
                "split": "test",
                "case_id": result["case_id"],
                "prediction": normalize(result["pred_report"]),
                "reference": normalize(result["reference_report"]),
                "source_file": str(path),
            })

    rows.sort(key=lambda row: (row["model"], row["case_id"]))
    if len(rows) != 368:
        raise ValueError(f"Expected 368 rows, produced {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with args.output.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Validated 4 feature-fusion models and 92 aligned cases per model")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
