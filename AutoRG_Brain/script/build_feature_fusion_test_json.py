#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build test_fusion_case_level.json for feature-level multimodal fusion.

This keeps the existing train_fusion_case_level.json training/validation splits
and appends a case-level test split from RadGenome-Brain_MRI.
"""

import argparse
import json
from pathlib import Path

from build_feature_fusion_train_json import (
    MODALITIES,
    case_has_required_files,
    disease_dir_for_case,
    get_target_text,
    load_json,
    load_report_sources,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radgenome_root",
        type=Path,
        default=Path("../../../RadGenome-Brain_MRI"),
        help="Path to RadGenome-Brain_MRI from AutoRG_Brain/script or an absolute path.",
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
        help="Path to Autorg_output containing autorg_output_test.",
    )
    parser.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Case-level split JSON. Defaults to RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json.",
    )
    parser.add_argument(
        "--train_json",
        type=Path,
        default=Path("../../raw_data/Task003_llm_fusion/train_fusion_case_level.json"),
        help="Existing feature-fusion train JSON with training/validation splits.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("../../raw_data/Task003_llm_fusion/test_fusion_case_level.json"),
        help="Output JSON containing training, validation, and test splits.",
    )
    parser.add_argument(
        "--anatomy_modal",
        choices=MODALITIES,
        default="t1c",
        help="Which predicted anatomy mask to require for test cases.",
    )
    parser.add_argument(
        "--region_value",
        choices=["abnormal", "global"],
        default="abnormal",
        help="Region selector stored in region_report.test.",
    )
    parser.add_argument(
        "--target_type",
        choices=["global", "modal_concat", "impression", "global_plus_impression"],
        default="global",
        help="Which RadGenome text to use as target report.",
    )
    parser.add_argument(
        "--no_require_images",
        action="store_true",
        help="Do not filter out cases missing BraTS_filtered image/seg/anatomy files.",
    )
    return parser.parse_args()


def resolve_from_script(script_dir, path):
    return (script_dir / path).resolve() if not path.is_absolute() else path


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    radgenome_root = resolve_from_script(script_dir, args.radgenome_root)
    brats_root = resolve_from_script(script_dir, args.brats_root)
    autorg_output_root = resolve_from_script(script_dir, args.autorg_output_root)
    train_json = resolve_from_script(script_dir, args.train_json)
    output_json = resolve_from_script(script_dir, args.output_json)
    split_json = args.split_json
    if split_json is None:
        split_json = radgenome_root / "train_val_test_case_level_split_GLI_MEN.json"
    else:
        split_json = resolve_from_script(script_dir, split_json)

    base = load_json(train_json)
    split_data = load_json(split_json)
    sources = load_report_sources(radgenome_root)
    require_images = not args.no_require_images

    output = {
        "training": base["training"],
        "validation": base["validation"],
        "test": [],
        "region_report": {
            "training": base["region_report"]["training"],
            "validation": base["region_report"]["validation"],
            "test": {},
        },
    }

    missing_records = []
    for case_id in split_data.get("test", []):
        if disease_dir_for_case(case_id) is None:
            missing_records.append({
                "split": "test",
                "case_id": case_id,
                "reason": "unsupported_disease",
            })
            continue

        text, report_error = get_target_text(case_id, args.target_type, sources)
        if report_error:
            missing_records.append({
                "split": "test",
                "case_id": case_id,
                "reason": report_error,
            })
            continue

        missing_files = case_has_required_files(
            brats_root,
            autorg_output_root,
            "test",
            case_id,
            args.anatomy_modal,
        ) if require_images else []
        if missing_files:
            missing_records.append({
                "split": "test",
                "case_id": case_id,
                "reason": "missing_image_or_seg",
                "missing": missing_files,
            })
            continue

        output["test"].append(case_id)
        output["region_report"]["test"][case_id] = {text: args.region_value}

    summary = {
        "target_type": args.target_type,
        "radgenome_root": str(radgenome_root),
        "brats_root": str(brats_root),
        "autorg_output_root": str(autorg_output_root),
        "anatomy_modal": args.anatomy_modal,
        "region_value": args.region_value,
        "split_json": str(split_json),
        "train_json": str(train_json),
        "require_images": require_images,
        "output_json": str(output_json),
        "splits": {
            "training": len(output["training"]),
            "validation": len(output["validation"]),
            "test_input": len(split_data.get("test", [])),
            "test_kept": len(output["test"]),
            "test_skipped": len(missing_records),
        },
    }

    output_dir = output_json.parent
    save_json(output, output_json)
    save_json(summary, output_dir / "test_fusion_case_level_summary.json")
    save_json(missing_records, output_dir / "test_fusion_case_level_missing_cases.json")

    print("\nFeature-fusion test JSON built.")
    print("Output JSON:", output_json)
    print("Summary:", output_dir / "test_fusion_case_level_summary.json")
    print("Missing cases:", output_dir / "test_fusion_case_level_missing_cases.json")
    print("\nCounts:")
    for key, value in summary["splits"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
