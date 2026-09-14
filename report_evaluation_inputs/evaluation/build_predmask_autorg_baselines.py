#!/usr/bin/env python3
"""Build Direct AutoRG and single-modal baselines from predicted-mask reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


MODAL_ORDER = ("t1n", "t2w", "t2f", "t1c")
IMAGE_PATTERN = re.compile(r"^(?P<case_id>.+)-(?P<modal>t1n|t1c|t2w|t2f)\.nii\.gz$")
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
MODEL_NAMES = {
    "t1n": "AutoRG-T1",
    "t1c": "AutoRG-T1C",
    "t2w": "AutoRG-T2",
    "t2f": "AutoRG-FLAIR",
}


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected report text to be str, got {type(value).__name__}")
    text = " ".join(value.split())
    if not text:
        raise ValueError("Encountered an empty generated report")
    return text


def image_path_from_record(record: dict[str, Any]) -> str:
    image = record.get("image")
    if isinstance(image, list):
        if len(image) != 1 or not isinstance(image[0], str):
            raise ValueError(f"Expected a one-item image list, got {image!r}")
        return image[0]
    if isinstance(image, str):
        return image
    raise TypeError(f"Unsupported image field: {image!r}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    evaluation_dir = Path(__file__).resolve().parent
    root = evaluation_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-report",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/autorg_output_test_predmask/pred_report.json"),
    )
    parser.add_argument(
        "--full-reference-input",
        type=Path,
        default=root / "feature_level" / "concat_projection_test.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=evaluation_dir / "prepared" / "predmask_autorg_baselines.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted = json.loads(args.pred_report.read_text(encoding="utf-8"))
    reference_payload = json.loads(args.full_reference_input.read_text(encoding="utf-8"))
    reference_rows = reference_payload.get("results")
    if not isinstance(reference_rows, list):
        raise ValueError("Expected full-reference input to contain a top-level results list")
    full_references = {
        row["case_id"]: normalize_text(row["reference_report"])
        for row in reference_rows
    }

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, record in enumerate(predicted):
        image_path = image_path_from_record(record)
        match = IMAGE_PATTERN.match(Path(image_path).name)
        if match is None:
            raise ValueError(f"Cannot parse case/modality from image path at record {index}: {image_path}")
        case_id = match.group("case_id")
        modal = match.group("modal")
        if modal in grouped[case_id]:
            raise ValueError(f"Duplicate predicted report for {case_id}/{modal}")
        grouped[case_id][modal] = record

    if set(grouped) != set(full_references):
        missing = sorted(set(full_references) - set(grouped))
        extra = sorted(set(grouped) - set(full_references))
        raise ValueError(f"Case mismatch. Missing={missing[:5]}, extra={extra[:5]}")
    for case_id, records in grouped.items():
        missing_modals = set(MODAL_ORDER) - set(records)
        if missing_modals:
            raise ValueError(f"Case {case_id} is missing modalities {sorted(missing_modals)}")

    output_rows: list[dict[str, str]] = []
    source_name = str(args.pred_report)
    for case_id in sorted(grouped):
        records = grouped[case_id]
        reference = full_references[case_id]
        output_rows.append({
            "category": "baseline",
            "model": "Direct-AutoRG",
            "input_source": "predicted_abnormal_mask",
            "split": "test",
            "case_id": case_id,
            "prediction": " ".join(normalize_text(records[m]["pred_report"]) for m in MODAL_ORDER),
            "reference": reference,
            "source_file": source_name,
        })
        for modal in MODAL_ORDER:
            output_rows.append({
                "category": "single_modal",
                "model": MODEL_NAMES[modal],
                "input_source": "predicted_abnormal_mask",
                "split": "test",
                "case_id": case_id,
                "prediction": normalize_text(records[modal]["pred_report"]),
                "reference": reference,
                "source_file": source_name,
            })

    output_rows.sort(key=lambda row: (row["category"], row["model"], row["case_id"]))
    if len(output_rows) != 5 * 92:
        raise ValueError(f"Expected 460 rows, produced {len(output_rows)}")

    write_csv(args.output_csv, output_rows)
    write_jsonl(args.output_csv.with_suffix(".jsonl"), output_rows)
    print(f"Validated {len(grouped)} cases and {len(predicted)} predicted-mask reports")
    print("Validated four modalities per case")
    print(f"Wrote 5 configurations / {len(output_rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
