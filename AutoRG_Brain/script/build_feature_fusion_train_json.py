# -*- coding: utf-8 -*-
"""
Build case-level training JSON for feature-level multimodal fusion.

This script aligns:
    - BraTS_filtered image folders
    - RadGenome-Brain_MRI case-level splits
    - RadGenome-Brain_MRI report targets

It creates the train file expected by train_llm_multi.py:
    {
        "training": [case_id, ...],
        "validation": [case_id, ...],
        "region_report": {
            "training": {case_id: {target_report: "abnormal"}},
            "validation": {case_id: {target_report: "abnormal"}}
        }
    }

The generated case IDs are case-level IDs such as:
    BraTS-GLI-00017-000

not modality-level IDs such as:
    BraTS-GLI-00017-000-t1n
"""

import argparse
import json
import shutil
from pathlib import Path


MODALITIES = ("t1n", "t1c", "t2w", "t2f")
DISEASE_DIRS = {
    "BraTS-GLI": "BraTS_GLI",
    "BraTS-MEN": "BraTS_MEN",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def disease_dir_for_case(case_id):
    for prefix, folder in DISEASE_DIRS.items():
        if case_id.startswith(prefix):
            return folder
    return None


def load_report_sources(radgenome_root):
    sources = {}
    for folder in DISEASE_DIRS.values():
        folder_path = radgenome_root / folder
        sources[folder] = {
            "global": load_json(folder_path / "global_finding.json"),
            "modal": load_json(folder_path / "modal_wise_finding.json"),
            "impression": load_json(folder_path / "impression.json"),
        }
    return sources


def get_target_text(case_id, target_type, sources):
    folder = disease_dir_for_case(case_id)
    if folder is None:
        return None, "unsupported_disease"

    source = sources[folder]

    if target_type == "global":
        text = source["global"].get(case_id)
        return text, None if text else "missing_global_report"

    if target_type == "impression":
        item = source["impression"].get(case_id)
        text = item.get("impression") if isinstance(item, dict) else None
        return text, None if text else "missing_impression"

    if target_type == "global_plus_impression":
        global_text = source["global"].get(case_id)
        item = source["impression"].get(case_id)
        impression = item.get("impression") if isinstance(item, dict) else None
        parts = [x for x in (global_text, impression) if x]
        return " ".join(parts), None if parts else "missing_global_plus_impression"

    if target_type == "modal_concat":
        parts = []
        missing = []
        for modal in MODALITIES:
            key = f"{case_id}-{modal}"
            text = source["modal"].get(key)
            if text:
                parts.append(text)
            else:
                missing.append(key)
        if missing:
            return None, "missing_modal_reports:" + ",".join(missing)
        return " ".join(parts), None

    raise ValueError(f"Unsupported target_type: {target_type}")


def find_case_images(brats_root, split, case_id):
    case_dir = brats_root / split / case_id
    image_paths = {modal: case_dir / f"{case_id}-{modal}.nii.gz" for modal in MODALITIES}
    seg_path = case_dir / f"{case_id}-seg.nii.gz"
    return case_dir, image_paths, seg_path


def find_anatomy_mask(autorg_output_root, split, case_id, anatomy_modal):
    return autorg_output_root / f"autorg_output_{split}" / f"{case_id}-{anatomy_modal}_ana.nii.gz"


def case_has_required_files(brats_root, autorg_output_root, split, case_id, anatomy_modal):
    case_dir, image_paths, seg_path = find_case_images(brats_root, split, case_id)
    ana_path = find_anatomy_mask(autorg_output_root, split, case_id, anatomy_modal)
    missing = []
    if not case_dir.is_dir():
        missing.append(str(case_dir))
    for path in image_paths.values():
        if not path.is_file():
            missing.append(str(path))
    if not seg_path.is_file():
        missing.append(str(seg_path))
    if not ana_path.is_file():
        missing.append(str(ana_path))
    return missing


def build_case_dic(training_cases, validation_cases):
    all_cases = sorted(set(training_cases) | set(validation_cases))

    # DataLoader3D_Multi still opens case_dic.json and expects these keys.
    # In multi-modal training selected_keys uses all cases, so duplicating
    # all case IDs across modality buckets is only a compatibility shim.
    return {
        "DWI": [],
        "T1WI": all_cases,
        "T2WI": all_cases,
        "T2FLAIR": all_cases,
    }


def build_dataset_json(brats_root, autorg_output_root, training_cases, validation_cases, anatomy_modal):
    entries = []
    split_by_case = {}
    for split, cases in (("train", training_cases), ("val", validation_cases)):
        for case_id in cases:
            split_by_case[case_id] = split

    for case_id in sorted(split_by_case):
        split = split_by_case[case_id]
        _, image_paths, seg_path = find_case_images(brats_root, split, case_id)
        ana_path = find_anatomy_mask(autorg_output_root, split, case_id, anatomy_modal)
        images = [str(image_paths[modal]) for modal in MODALITIES]
        entries.append({
            "image": images[0],
            "images": images,
            "label1": str(ana_path),
            "label2": str(seg_path),
            "modal": "multi",
        })

    labels = {str(i): str(i) for i in range(96)}
    return {
        "name": "Task003_llm_fusion",
        "description": "Feature-level multimodal AutoRG-Brain fusion with T1C anatomy mask and GT abnormal mask.",
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
        help="Path to Autorg_output containing autorg_output_train/autorg_output_val.",
    )
    parser.add_argument(
        "--anatomy_modal",
        choices=MODALITIES,
        default="t1c",
        help="Which predicted anatomy mask to use as fixed anatomy supervision.",
    )
    parser.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Case-level split JSON. Defaults to RadGenome-Brain_MRI/train_val_test_case_level_split_GLI_MEN.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../../raw_data/Task003_llm_fusion"),
        help="Output directory for generated JSON files.",
    )
    parser.add_argument(
        "--nnunet_raw_output_dir",
        type=Path,
        default=Path("../../raw_data/nnUNet_raw_data/Task003_llm_fusion"),
        help="Output directory for nnU-Net dataset.json.",
    )
    parser.add_argument(
        "--region_value",
        choices=["abnormal", "global"],
        default="abnormal",
        help="Region selector stored in train_fusion_case_level.json.",
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
        help="Do not filter out cases missing BraTS_filtered image/seg files.",
    )
    parser.add_argument(
        "--copy_case_dic_to",
        type=Path,
        default=Path("../case_dic.json"),
        help="Also copy generated case_dic.json here. Use '' to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    radgenome_root = (script_dir / args.radgenome_root).resolve() if not args.radgenome_root.is_absolute() else args.radgenome_root
    brats_root = (script_dir / args.brats_root).resolve() if not args.brats_root.is_absolute() else args.brats_root
    autorg_output_root = (script_dir / args.autorg_output_root).resolve() if not args.autorg_output_root.is_absolute() else args.autorg_output_root
    output_dir = (script_dir / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    nnunet_raw_output_dir = (script_dir / args.nnunet_raw_output_dir).resolve() if not args.nnunet_raw_output_dir.is_absolute() else args.nnunet_raw_output_dir

    split_json = args.split_json
    if split_json is None:
        split_json = radgenome_root / "train_val_test_case_level_split_GLI_MEN.json"
    elif not split_json.is_absolute():
        split_json = (script_dir / split_json).resolve()

    split_data = load_json(split_json)
    sources = load_report_sources(radgenome_root)

    require_images = not args.no_require_images
    output = {
        "training": [],
        "validation": [],
        "region_report": {
            "training": {},
            "validation": {},
        },
    }
    missing_records = []
    summary = {
        "target_type": args.target_type,
        "radgenome_root": str(radgenome_root),
        "brats_root": str(brats_root),
        "autorg_output_root": str(autorg_output_root),
        "anatomy_modal": args.anatomy_modal,
        "region_value": args.region_value,
        "split_json": str(split_json),
        "require_images": require_images,
        "splits": {},
    }

    split_name_map = {
        "train": ("training", "training"),
        "val": ("validation", "validation"),
    }

    for source_split, (list_key, report_key) in split_name_map.items():
        input_cases = split_data.get(source_split, [])
        kept = []
        skipped = 0

        for case_id in input_cases:
            if disease_dir_for_case(case_id) is None:
                skipped += 1
                missing_records.append({
                    "split": source_split,
                    "case_id": case_id,
                    "reason": "unsupported_disease",
                })
                continue

            text, report_error = get_target_text(case_id, args.target_type, sources)
            if report_error:
                skipped += 1
                missing_records.append({
                    "split": source_split,
                    "case_id": case_id,
                    "reason": report_error,
                })
                continue

            missing_files = case_has_required_files(
                brats_root,
                autorg_output_root,
                source_split,
                case_id,
                args.anatomy_modal,
            ) if require_images else []
            if missing_files:
                skipped += 1
                missing_records.append({
                    "split": source_split,
                    "case_id": case_id,
                    "reason": "missing_image_or_seg",
                    "missing": missing_files,
                })
                continue

            kept.append(case_id)
            output["region_report"][report_key][case_id] = {text: args.region_value}

        output[list_key] = kept
        summary["splits"][source_split] = {
            "input_cases": len(input_cases),
            "kept_cases": len(kept),
            "skipped_cases": skipped,
        }

    output_path = output_dir / "train_fusion_case_level.json"
    case_dic_path = output_dir / "case_dic.json"
    summary_path = output_dir / "summary.json"
    missing_path = output_dir / "missing_cases.json"
    dataset_json_path = nnunet_raw_output_dir / "dataset.json"

    save_json(output, output_path)
    save_json(build_case_dic(output["training"], output["validation"]), case_dic_path)
    save_json(summary, summary_path)
    save_json(missing_records, missing_path)
    save_json(
        build_dataset_json(
            brats_root,
            autorg_output_root,
            output["training"],
            output["validation"],
            args.anatomy_modal,
        ),
        dataset_json_path,
    )

    copy_case_dic_to = args.copy_case_dic_to
    if str(copy_case_dic_to):
        if not copy_case_dic_to.is_absolute():
            copy_case_dic_to = (script_dir / copy_case_dic_to).resolve()
        copy_case_dic_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(case_dic_path, copy_case_dic_to)

    print("\nFeature-fusion training JSON built.")
    print("Output train JSON:", output_path)
    print("Output nnU-Net dataset.json:", dataset_json_path)
    print("Output case_dic:", case_dic_path)
    if str(args.copy_case_dic_to):
        print("Copied case_dic to:", copy_case_dic_to)
    print("Summary:", summary_path)
    print("Missing cases:", missing_path)
    print("\nCounts:")
    for split, stats in summary["splits"].items():
        print(
            f"  {split}: input={stats['input_cases']} "
            f"kept={stats['kept_cases']} skipped={stats['skipped_cases']}"
        )


if __name__ == "__main__":
    main()
