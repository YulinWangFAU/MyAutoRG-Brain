#!/usr/bin/env python3
"""Build the 92-case text-fusion test set from predicted-mask AutoRG reports."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


MODAL_ORDER = ("t1n", "t2w", "t2f", "t1c")
IMAGE_PATTERN = re.compile(r"^(?P<case_id>.+)-(?P<modal>t1n|t1c|t2w|t2f)\.nii\.gz$")


def normalize(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected text, got {type(value).__name__}")
    text = " ".join(value.split())
    if not text:
        raise ValueError("Encountered an empty report")
    return text


def image_path(record: dict[str, Any]) -> str:
    value = record.get("image")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported image field: {value!r}")


def build_prompt(modals: dict[str, str]) -> str:
    return f"""You are a radiology expert.

Below are modality-specific MRI findings for the same patient.

[T1-weighted Imaging]
{modals['t1n']}

[T2-weighted Imaging]
{modals['t2w']}

[FLAIR Imaging]
{modals['t2f']}

[T1CE Imaging]
{modals['t1c']}

Please integrate the above findings into a comprehensive radiology report for this patient."""


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent
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
        "--output",
        type=Path,
        default=here / "prepared" / "late_fusion_test_predmask.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = json.loads(args.pred_report.read_text(encoding="utf-8"))
    reference_payload = json.loads(args.full_reference_input.read_text(encoding="utf-8"))
    reference_rows = reference_payload.get("results")
    if not isinstance(reference_rows, list):
        raise ValueError("Expected a top-level 'results' list in reference input")
    references = {
        row["case_id"]: normalize(row["reference_report"])
        for row in reference_rows
    }

    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for index, record in enumerate(predictions):
        match = IMAGE_PATTERN.match(Path(image_path(record)).name)
        if match is None:
            raise ValueError(f"Cannot parse case/modality at record {index}")
        case_id, modal = match.group("case_id"), match.group("modal")
        if modal in grouped[case_id]:
            raise ValueError(f"Duplicate report for {case_id}/{modal}")
        grouped[case_id][modal] = normalize(record["pred_report"])

    if set(grouped) != set(references):
        missing = sorted(set(references) - set(grouped))
        extra = sorted(set(grouped) - set(references))
        raise ValueError(f"Case mismatch. Missing={missing[:5]}, extra={extra[:5]}")

    dataset = []
    for case_id in sorted(grouped):
        missing_modals = set(MODAL_ORDER) - set(grouped[case_id])
        if missing_modals:
            raise ValueError(f"{case_id} missing modalities: {sorted(missing_modals)}")
        dataset.append({
            "case_id": case_id,
            "input_text": build_prompt(grouped[case_id]),
            "target_text": references[case_id],
            "input_source": "predicted_abnormal_mask",
        })

    if len(dataset) != 92:
        raise ValueError(f"Expected 92 cases, produced {len(dataset)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Validated {len(predictions)} reports / {len(dataset)} cases / four modalities per case")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
