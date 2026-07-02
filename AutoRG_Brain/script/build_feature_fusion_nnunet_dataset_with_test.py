#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build an nnU-Net dataset.json that includes train, validation, and test cases.

The feature-fusion preprocessing path reads dataset.json["training"] only, so
test cases must be listed as training entries if we want them to be converted
into preprocessed .npz/.pkl files for report inference.
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


def case_entry(brats_root, autorg_output_root, split, case_id, anatomy_modal):
    case_dir = brats_root / split / case_id
    images = [case_dir / f"{case_id}-{mod}.nii.gz" for mod in MODALITIES]
    seg = case_dir / f"{case_id}-seg.nii.gz"
    ana = autorg_output_root / f"autorg_output_{split}" / f"{case_id}-{anatomy_modal}_ana.nii.gz"

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
        help="JSON containing training, validation, and test case IDs.",
    )
    parser.add_argument(
        "--brats_root",
        type=Path,
        default=Path("../../../BraTS_filtered"),
        help="Path to BraTS_filtered from AutoRG_Brain/script or an absolute path.",
    )
    parser.add_argument(
        "--autorg_output_root",
        type=Path,
        default=Path("../../../Autorg_output"),
        help="Path to Autorg_output containing autorg_output_train/val/test.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("../../raw_data/nnUNet_raw_data/Task003_llm_fusion/dataset_with_test.json"),
        help="Output nnU-Net dataset JSON. Copy this to dataset.json before preprocessing.",
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
        help="Write dataset JSON even if some image/label paths are missing.",
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
    split_map = (
        ("train", "training"),
        ("val", "validation"),
        ("test", "test"),
    )

    entries = []
    missing_records = []
    for raw_split, json_split in split_map:
        for case_id in split_data.get(json_split, []):
            entry, missing = case_entry(
                brats_root,
                autorg_output_root,
                raw_split,
                case_id,
                args.anatomy_modal,
            )
            if missing:
                missing_records.append({
                    "split": raw_split,
                    "case_id": case_id,
                    "missing": missing,
                })
                if not args.no_require_files:
                    continue
            entries.append(entry)

    labels = {str(i): str(i) for i in range(96)}
    dataset_json = {
        "name": "Task003_llm_fusion",
        "description": "Feature-level multimodal AutoRG-Brain fusion with train/val/test cases for preprocessing.",
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
    missing_path = output_json.parent / "dataset_with_test_missing_files.json"
    save_json(missing_records, missing_path)

    print("\nFeature-fusion nnU-Net dataset JSON built.")
    print("Output dataset JSON:", output_json)
    print("Missing files JSON:", missing_path)
    print("\nCounts:")
    print("  training cases:", len(split_data.get("training", [])))
    print("  validation cases:", len(split_data.get("validation", [])))
    print("  test cases:", len(split_data.get("test", [])))
    print("  written entries:", len(entries))
    print("  cases with missing files:", len(missing_records))


if __name__ == "__main__":
    main()
