#!/usr/bin/env python3
"""Build aligned Direct AutoRG and single-modality case-level baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


MODAL_ORDER = ("t1n", "t2w", "t2f", "t1c")
MODAL_NAMES = {
    "t1n": "T1",
    "t1c": "T1C",
    "t2w": "T2",
    "t2f": "FLAIR",
}


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected text, got {type(value).__name__}")
    return " ".join(value.split())


def load_records(path: Path, container: str | None = None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if container is not None:
        payload = payload[container]
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list in {path}")
    return payload


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    evaluation_dir = Path(__file__).resolve().parent
    input_root = evaluation_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single-modal-input",
        type=Path,
        default=input_root / "single_modal" / "autorg_single_modal_test_with_gt.json",
    )
    parser.add_argument(
        "--full-reference-input",
        type=Path,
        default=input_root / "feature_level" / "concat_projection_test.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=input_root / "single_modal" / "case_level_baselines",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_records(args.single_modal_input)
    full_reference_rows = load_records(args.full_reference_input, container="results")

    full_references = {
        row["case_id"]: normalize_text(row["reference_report"])
        for row in full_reference_rows
    }
    if len(full_references) != 92:
        raise ValueError(f"Expected 92 full references, found {len(full_references)}")

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in source:
        case_id = row.get("case_id")
        modal = row.get("modal")
        if case_id not in full_references:
            raise ValueError(f"No full reference for case {case_id!r}")
        if modal not in MODAL_ORDER:
            raise ValueError(f"Unexpected modality {modal!r} for case {case_id!r}")
        if modal in grouped[case_id]:
            raise ValueError(f"Duplicate {modal!r} report for case {case_id!r}")
        grouped[case_id][modal] = row

    if set(grouped) != set(full_references):
        raise ValueError("Single-modal cases do not align with full-reference cases")
    for case_id, modal_rows in grouped.items():
        missing = set(MODAL_ORDER) - set(modal_rows)
        if missing:
            raise ValueError(f"Case {case_id!r} is missing modalities: {sorted(missing)}")

    direct_rows = []
    single_rows: dict[str, list[dict]] = {modal: [] for modal in MODAL_ORDER}
    for case_id in sorted(grouped):
        modal_rows = grouped[case_id]
        full_reference = full_references[case_id]
        direct_rows.append({
            "case_id": case_id,
            "prediction": " ".join(normalize_text(modal_rows[m]["prediction"]) for m in MODAL_ORDER),
            "reference": full_reference,
            "modal_order": list(MODAL_ORDER),
        })
        for modal in MODAL_ORDER:
            single_rows[modal].append({
                "case_id": case_id,
                "prediction": normalize_text(modal_rows[modal]["prediction"]),
                "reference": full_reference,
                "modality_reference": normalize_text(modal_rows[modal]["reference"]),
                "modal": modal,
            })

    write_json(args.output_dir / "direct_autorg_test_with_full_gt.json", direct_rows)
    for modal, records in single_rows.items():
        write_json(args.output_dir / f"autorg_{modal}_test_with_full_gt.json", records)

    print(f"Validated {len(grouped)} cases with four modalities each")
    print(f"Wrote 5 case-level baseline files to {args.output_dir}")
    print("Direct AutoRG concatenation order:", " -> ".join(MODAL_NAMES[m] for m in MODAL_ORDER))


if __name__ == "__main__":
    main()
