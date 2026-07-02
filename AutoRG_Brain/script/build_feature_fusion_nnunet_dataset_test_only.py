#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a test-only nnU-Net dataset.json for feature-level fusion preprocessing.

The feature-fusion preprocessing code reads dataset.json["training"], so this
file intentionally writes test cases into the "training" list. Use it only as a
temporary dataset.json while preprocessing test cases.
"""

import argparse
import json
from pathlib import Path


MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def resolve_from_script(script_dir, path):
    path = Path(path)
    return (script_dir / path).resolve() if not path.is_absolute() else path


def build_entry(brats_root, autorg_output_root, case_id, anatomy_modal):
    case_dir = brats_root / "test" / case_id
    images = [case_dir / f"{case_id}-{mod}.nii.gz" for mod in MODALITIES]
    seg = case_dir / f"{case_id}-seg.nii.gz"
    ana = autorg_output_root / "autorg_output_test" / f"{case_id}-{anatomy_modal}_ana.nii.gz"

    missing = [str(path) for path in images + [seg, ana] if not path.is_file()]
    entry = {
        "image": str(images[0]),
        "images": [str(path) for path in images],
        "label1": str(ana),
        "label2": str(seg),
        "modal": "multi",
    }
    return entry, missing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path("../../raw_data/Task003_llm_fusion/test_fusion_case_level.json"),
        help="JSON containing the test case IDs under key 'test'.",
    )
    parser.add_argument(
        "--brats_root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/BraTS_filtered"),
        help="Root containing train/val/test BraTS_filtered folders.",
    )
    parser.add_argument(
        "--autorg_output_root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/Autorg_output"),
        help="Root containing autorg_output_train/val/test folders.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("../../raw_data/nnUNet_raw_data/Task003_llm_fusion/dataset_test_only.json"),
        help="Output test-only nnU-Net dataset JSON.",
    )
    parser.add_argument(
        "--anatomy_modal",
        choices=MODALITIES,
        default="t1c",
        help="Which predicted anatomy mask to use as label1.",
    )
    parser.add_argument(
        "--no_require_files",
        action="store_true",
        help="Write entries even if some image/label paths are missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    train_file = resolve_from_script(script_dir, args.train_file)
    brats_root = resolve_from_script(script_dir, args.brats_root)
    autorg_output_root = resolve_from_script(script_dir, args.autorg_output_root)
    output_json = resolve_from_script(script_dir, args.output_json)

    split_data = load_json(train_file)
    entries = []
    missing_records = []

    for case_id in split_data.get("test", []):
        entry, missing = build_entry(brats_root, autorg_output_root, case_id, args.anatomy_modal)
        if missing:
            missing_records.append({
                "case_id": case_id,
                "missing": missing,
            })
            if not args.no_require_files:
                continue
        entries.append(entry)

    labels = {str(i): str(i) for i in range(96)}
    dataset_json = {
        "name": "Task003_llm_fusion",
        "description": "Test-only feature-level multimodal AutoRG-Brain fusion preprocessing dataset.",
        "tensorImageSize": "4D",
        "reference": "RadGenome-Brain_MRI + BraTS_filtered + AutoRG_output",
        "licence": "research",
        "release": "0.0",
        "modality": {
            "0": "T1N",
            "1": "T1C",
            "2": "T2W",
            "3": "T2F",
        },
        "labels": labels,
        "numTraining": len(entries),
        "training": entries,
        "test": [],
    }

    save_json(dataset_json, output_json)
    missing_path = output_json.parent / "dataset_test_only_missing_files.json"
    save_json(missing_records, missing_path)

    print("\nFeature-fusion test-only nnU-Net dataset JSON built.")
    print("Output dataset JSON:", output_json)
    print("Missing files JSON:", missing_path)
    print("BraTS root:", brats_root)
    print("Autorg output root:", autorg_output_root)
    print("\nCounts:")
    print("  test cases:", len(split_data.get("test", [])))
    print("  written entries:", len(entries))
    print("  cases with missing files:", len(missing_records))


if __name__ == "__main__":
    main()
