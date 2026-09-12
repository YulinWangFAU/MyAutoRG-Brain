#!/usr/bin/env python3
"""Build one case-level evaluation table from all report-generation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportSource:
    category: str
    model: str
    input_source: str
    relative_path: str
    container: str = "list"
    prediction_field: str = "prediction"
    reference_field: str = "reference"


SOURCES = (
    ReportSource(
        "baseline",
        "Direct-AutoRG",
        "gt_abnormal_mask",
        "single_modal/case_level_baselines/direct_autorg_test_with_full_gt.json",
    ),
    ReportSource(
        "single_modal",
        "AutoRG-T1",
        "gt_abnormal_mask",
        "single_modal/case_level_baselines/autorg_t1n_test_with_full_gt.json",
    ),
    ReportSource(
        "single_modal",
        "AutoRG-T1C",
        "gt_abnormal_mask",
        "single_modal/case_level_baselines/autorg_t1c_test_with_full_gt.json",
    ),
    ReportSource(
        "single_modal",
        "AutoRG-T2",
        "gt_abnormal_mask",
        "single_modal/case_level_baselines/autorg_t2w_test_with_full_gt.json",
    ),
    ReportSource(
        "single_modal",
        "AutoRG-FLAIR",
        "gt_abnormal_mask",
        "single_modal/case_level_baselines/autorg_t2f_test_with_full_gt.json",
    ),
    ReportSource(
        "text_level",
        "T5-Small",
        "autorg_prediction",
        "text_level/autorg_input/t5_small_test.json",
    ),
    ReportSource(
        "text_level",
        "Flan-T5-Large-LoRA",
        "autorg_prediction",
        "text_level/autorg_input/flan_t5_large_lora_test.json",
    ),
    ReportSource(
        "text_level",
        "Flan-T5-Large-LoRA-PullLoss",
        "autorg_prediction",
        "text_level/autorg_input/flan_t5_large_lora_pullloss_test.json",
    ),
    ReportSource(
        "text_level",
        "T5-Small",
        "oracle_ground_truth",
        "text_level/oracle_gt_input/t5_small_test.json",
    ),
    ReportSource(
        "text_level",
        "Flan-T5-Large-LoRA",
        "oracle_ground_truth",
        "text_level/oracle_gt_input/flan_t5_large_lora_test.json",
    ),
    ReportSource(
        "text_level",
        "Flan-T5-Large-LoRA-PullLoss",
        "oracle_ground_truth",
        "text_level/oracle_gt_input/flan_t5_large_lora_pullloss_test.json",
    ),
    ReportSource(
        "feature_level",
        "Mean",
        "image_features",
        "feature_level/mean_test.json",
        container="results",
        prediction_field="pred_report",
        reference_field="reference_report",
    ),
    ReportSource(
        "feature_level",
        "Weighted-Mean",
        "image_features",
        "feature_level/weighted_mean_test.json",
        container="results",
        prediction_field="pred_report",
        reference_field="reference_report",
    ),
    ReportSource(
        "feature_level",
        "Token-Gated",
        "image_features",
        "feature_level/token_gated_test.json",
        container="results",
        prediction_field="pred_report",
        reference_field="reference_report",
    ),
    ReportSource(
        "feature_level",
        "Concat-Projection",
        "image_features",
        "feature_level/concat_projection_test.json",
        container="results",
        prediction_field="pred_report",
        reference_field="reference_report",
    ),
    ReportSource(
        "vlm",
        "Qwen2.5-VL-Zero-Shot",
        "image_montage",
        "vlm/qwen25vl_zero_shot_test_with_gt.json",
    ),
    ReportSource(
        "vlm",
        "Qwen2.5-VL-Few-Shot",
        "image_montage",
        "vlm/qwen25vl_few_shot_test_with_gt.json",
    ),
)

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


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected report text to be str, got {type(value).__name__}")
    return " ".join(value.split())


def load_source(root: Path, source: ReportSource) -> list[dict[str, str]]:
    path = root / source.relative_path
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if source.container == "list":
        records = payload
    elif source.container == "results":
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"Expected a top-level results list in {path}")
        records = payload["results"]
    else:
        raise ValueError(f"Unsupported container: {source.container}")

    if not isinstance(records, list):
        raise TypeError(f"Expected a list of records in {path}")

    output = []
    seen_case_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record {index} in {path} is not an object")

        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Missing case_id in record {index} of {path}")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id {case_id!r} in {path}")
        seen_case_ids.add(case_id)

        output.append(
            {
                "category": source.category,
                "model": source.model,
                "input_source": source.input_source,
                "split": "test",
                "case_id": case_id,
                "prediction": normalize_text(record[source.prediction_field]),
                "reference": normalize_text(record[source.reference_field]),
                "source_file": source.relative_path,
            }
        )

    return output


def validate_alignment(groups: list[list[dict[str, str]]]) -> None:
    if not groups:
        raise ValueError("No report groups were loaded")

    expected_ids = {row["case_id"] for row in groups[0]}
    expected_refs = {row["case_id"]: row["reference"] for row in groups[0]}

    for rows in groups:
        ids = {row["case_id"] for row in rows}
        refs = {row["case_id"]: row["reference"] for row in rows}
        if ids != expected_ids:
            missing = sorted(expected_ids - ids)
            extra = sorted(ids - expected_ids)
            raise ValueError(f"Case-ID mismatch. Missing={missing[:5]}, extra={extra[:5]}")
        if refs != expected_refs:
            mismatched = sorted(case_id for case_id in expected_ids if refs[case_id] != expected_refs[case_id])
            raise ValueError(f"Reference mismatch for case IDs: {mismatched[:10]}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "prepared")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = args.input_root / "single_modal" / "case_level_baselines"
    if not baseline_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {baseline_dir}. Run build_autorg_baselines.py before build_unified_dataset.py."
        )
    groups = [load_source(args.input_root, source) for source in SOURCES]
    validate_alignment(groups)

    rows = [row for group in groups for row in group]
    rows.sort(key=lambda row: (row["category"], row["input_source"], row["model"], row["case_id"]))

    csv_path = args.output_dir / "unified_case_level_reports.csv"
    jsonl_path = args.output_dir / "unified_case_level_reports.jsonl"
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)

    print(f"Validated {len(SOURCES)} model/input configurations")
    print(f"Validated {len(groups[0])} aligned test cases per configuration")
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {len(rows)} rows to {jsonl_path}")


if __name__ == "__main__":
    main()
