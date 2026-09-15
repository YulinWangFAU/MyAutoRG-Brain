#!/usr/bin/env python3
"""Build isolated Task004 test data for predicted-mask feature fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-split-json",
        type=Path,
        default=root.parent / "raw_data" / "Task003_llm_fusion" /
        "fusion_case_level_train_val_test_abnormal.json",
    )
    parser.add_argument(
        "--brats-root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/BraTS_filtered/test"),
    )
    parser.add_argument(
        "--anatomy-root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/autorg_output_test"),
    )
    parser.add_argument(
        "--abnormal-root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/autorg_output_test_predmask"),
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=root.parent / "raw_data" / "nnUNet_raw_data" /
        "Task004_llm_fusion_predmask" / "dataset.json",
    )
    parser.add_argument(
        "--inference-json",
        type=Path,
        default=root.parent / "raw_data" / "Task004_llm_fusion_predmask" /
        "fusion_predmask_inference.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_json(args.source_split_json)
    test_cases = source.get("test")
    reports = source.get("region_report", {}).get("test")
    if not isinstance(test_cases, list) or len(test_cases) != 92:
        raise ValueError(f"Expected 92 source test cases, got {type(test_cases).__name__}")
    if not isinstance(reports, dict):
        raise ValueError("Source JSON is missing region_report/test")
    if set(test_cases) != set(reports):
        raise ValueError("Test cases and region_report/test keys do not match")

    entries = []
    missing: list[str] = []
    for case_id in test_cases:
        case_dir = args.brats_root / case_id
        images = [case_dir / f"{case_id}-{modal}.nii.gz" for modal in MODALITIES]
        anatomy = args.anatomy_root / f"{case_id}-t1c_ana.nii.gz"
        abnormal = args.abnormal_root / f"{case_id}-t1c_ab.nii.gz"
        paths = images + [anatomy, abnormal]
        missing.extend(str(path) for path in paths if not path.is_file())
        entries.append({
            "image": str(images[0]),
            "images": [str(path) for path in images],
            "label1": str(anatomy),
            "label2": str(abnormal),
            "modal": "multi",
        })
    if missing:
        raise FileNotFoundError("Missing A3 inputs:\n" + "\n".join(missing[:20]))

    dataset = {
        "name": "Task004_llm_fusion_predmask",
        "description": "Test-only feature fusion with predicted T1C abnormal masks.",
        "tensorImageSize": "4D",
        "reference": "BraTS_filtered + AutoRG predicted anatomy/abnormal masks",
        "licence": "research",
        "release": "0.0",
        "modality": {"0": "T1N", "1": "T1C", "2": "T2W", "3": "T2F"},
        "labels": {str(index): str(index) for index in range(96)},
        "numTraining": len(entries),
        "training": entries,
        "test": [],
    }
    # No optimization occurs during inference. Repeating the same 92 test IDs in
    # these split fields only satisfies the legacy trainer's loader initialization.
    inference = {
        "training": list(test_cases),
        "validation": list(test_cases),
        "test": list(test_cases),
        "region_report": {
            "training": reports,
            "validation": reports,
            "test": reports,
        },
    }
    write_json(args.dataset_json, dataset)
    write_json(args.inference_json, inference)
    print("Validated 92 cases with four MRI modalities each")
    print("label1: predicted T1C anatomy mask")
    print("label2: predicted T1C abnormal mask")
    print(f"Wrote dataset: {args.dataset_json}")
    print(f"Wrote inference splits: {args.inference_json}")


if __name__ == "__main__":
    main()
