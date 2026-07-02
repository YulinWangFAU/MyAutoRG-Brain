#!/usr/bin/env python
import argparse
import json
import pickle
from pathlib import Path


def load_json(path):
    with Path(path).open("r") as f:
        return json.load(f)


def load_plans(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def normalize_split_name(name):
    if name == "train":
        return "training"
    if name == "val":
        return "validation"
    return name


def case_exists(preprocessed_stage_dir, case_id):
    candidates = [
        preprocessed_stage_dir / f"{case_id}.npz",
        preprocessed_stage_dir / f"{case_id}.npy",
        preprocessed_stage_dir / f"{case_id}.pkl",
    ]
    return any(path.exists() for path in candidates)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether feature-fusion JSON cases exist in the nnU-Net preprocessed folder."
    )
    parser.add_argument(
        "--train_file",
        default="../raw_data/Task003_llm_fusion/test_fusion_case_level.json",
        help="Feature-fusion train/test JSON containing training, validation, and/or test keys.",
    )
    parser.add_argument(
        "--plans_file",
        default="utils_file/nnUNetPlansv2.1_plans_3D.pkl",
        help="nnU-Net plans file used by training/inference.",
    )
    parser.add_argument(
        "--preprocessed_root",
        default="../preprocessed_data",
        help="Root folder that contains Task003_llm_fusion.",
    )
    parser.add_argument(
        "--task",
        default="Task003_llm_fusion",
        help="Task folder name under --preprocessed_root.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="nnU-Net stage. If omitted, use the last stage in plans, same as 3d_fullres training.",
    )
    parser.add_argument(
        "--max_show",
        type=int,
        default=20,
        help="Maximum number of missing case IDs to print per split.",
    )
    args = parser.parse_args()

    train_file = Path(args.train_file)
    plans_file = Path(args.plans_file)
    preprocessed_root = Path(args.preprocessed_root)

    data = load_json(train_file)
    plans = load_plans(plans_file)
    data_identifier = plans["data_identifier"]
    stage = args.stage
    if stage is None:
        stage = sorted(plans["plans_per_stage"].keys())[-1]

    task_dir = preprocessed_root / args.task
    stage_dir = task_dir / f"{data_identifier}_stage{stage}"

    print(f"train_file: {train_file.resolve()}")
    print(f"plans_file: {plans_file.resolve()}")
    print(f"task_dir: {task_dir.resolve()}")
    print(f"preprocessed_stage_dir: {stage_dir.resolve()}")
    print(f"preprocessed_stage_dir_exists: {stage_dir.exists()}")
    print()

    if not stage_dir.exists():
        raise SystemExit("Preprocessed stage folder does not exist.")

    for requested_name in ("training", "validation", "test"):
        split_name = normalize_split_name(requested_name)
        cases = data.get(split_name, [])
        missing = [case_id for case_id in cases if not case_exists(stage_dir, case_id)]
        present = len(cases) - len(missing)
        print(f"{split_name}: total={len(cases)} present={present} missing={len(missing)}")
        if missing:
            shown = ", ".join(missing[: args.max_show])
            suffix = " ..." if len(missing) > args.max_show else ""
            print(f"  missing_examples: {shown}{suffix}")


if __name__ == "__main__":
    main()
