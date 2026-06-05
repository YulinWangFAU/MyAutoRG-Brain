# -*- coding: utf-8 -*-
"""
Early-fusion inference script for AutoRG-Brain.

Purpose
-------
Use the original AutoRG-Brain inference pipeline, but replace:

    network.language_model_patchwise

with:

    network.language_model_patchwise_earlyfusion

before importing AutoRG_Brain.

This allows us to save image_hidden_states before the GPT decoder,
without modifying the original project files.

Author: Yulin Wang
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


# =========================================================
# Important:
# Before importing AutoRG_Brain, replace the original
# network.language_model_patchwise with the early-fusion version.
#
# This avoids changing other import statements in the project.
# =========================================================

early_lm = importlib.import_module("network.language_model_patchwise_earlyfusion")
sys.modules["network.language_model_patchwise"] = early_lm

print("\n[EARLY FUSION] Module replacement finished.")
print("[EARLY FUSION] network.language_model_patchwise -> network.language_model_patchwise_earlyfusion")
print("[EARLY FUSION] Loaded module:", early_lm.__name__)


from inference.inferenceSdk import AutoRG_Brain


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--out_dir",
        required=True,
        default=None,
        help="folder for saving mask and report predictions"
    )

    parser.add_argument(
        "--llm_folder",
        required=True,
        default=None,
        help="folder of saved llm checkpoint"
    )

    parser.add_argument(
        "--llm_chk",
        help="llm checkpoint name",
        default="AutoRG_Brain_RGv1"
    )

    parser.add_argument(
        "--seg_folder",
        required=True,
        default=None,
        help="folder of saved segmentation checkpoint"
    )

    parser.add_argument(
        "--seg_chk",
        help="segmentation checkpoint name",
        default="AutoRG_Brain_SEG"
    )

    parser.add_argument(
        "-test",
        "--test_file",
        required=True,
        default=None,
        help="json with your test images info"
    )

    parser.add_argument(
        "--eval_mode",
        required=False,
        default="region_segtool",
        help="the report inference way"
    )

    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="save image_hidden_states before GPT decoder"
    )

    parser.add_argument(
        "--feature_dir",
        default=None,
        help="folder for saving intermediate image_hidden_states"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_file = Path(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Cannot find test_file: {test_file}")

    # =========================================================
    # Set environment variables for saving intermediate features
    # =========================================================

    if args.save_intermediate:
        if args.feature_dir is None:
            args.feature_dir = str(out_dir / "features")

        feature_dir = Path(args.feature_dir)
        feature_dir.mkdir(parents=True, exist_ok=True)

        os.environ["AUTORG_SAVE_INTERMEDIATE"] = "1"
        os.environ["AUTORG_FEATURE_DIR"] = str(feature_dir)

        print("\n[EARLY FUSION] save_intermediate enabled")
        print("[EARLY FUSION] feature_dir:", feature_dir.resolve())

    else:
        os.environ["AUTORG_SAVE_INTERMEDIATE"] = "0"
        print("\n[EARLY FUSION] save_intermediate disabled")

    print("[EARLY FUSION] AUTORG_SAVE_INTERMEDIATE:", os.environ.get("AUTORG_SAVE_INTERMEDIATE"))
    print("[EARLY FUSION] AUTORG_FEATURE_DIR:", os.environ.get("AUTORG_FEATURE_DIR"))

    # =========================================================
    # AutoRG-Brain config
    # =========================================================

    config = {
        "llm_folder": args.llm_folder,
        "seg_folder": args.seg_folder,
        "llm_chk": args.llm_chk,
        "seg_chk": args.seg_chk,
        "output_dir": str(out_dir),
        "eval_mode": args.eval_mode,
    }

    print("\n[EARLY FUSION] Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # =========================================================
    # Initialize AutoRG-Brain model
    # =========================================================

    model = AutoRG_Brain(gpu_id=[0], config=config)

    # =========================================================
    # Load test file
    # =========================================================

    with open(test_file, "r", encoding="utf-8") as f:
        input_case_dict = json.load(f)

    print("\n[EARLY FUSION] Loaded test cases from:", test_file)
    print("[EARLY FUSION] Number of input items:", len(input_case_dict))

    # =========================================================
    # Run report generation
    # =========================================================

    results = model.report(input_case_dict)

    pred_path = out_dir / "pred_report.json"

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n[EARLY FUSION] Finished inference.")
    print("[EARLY FUSION] pred_report saved to:", pred_path.resolve())

    if args.save_intermediate:
        print("[EARLY FUSION] features saved to:", Path(args.feature_dir).resolve())


if __name__ == "__main__":
    main()