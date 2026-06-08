# -*- coding: utf-8 -*-
"""
Fused-feature inference script for AutoRG-Brain.

Purpose
-------
Use the original AutoRG-Brain inference pipeline, but replace:

    network.language_model_patchwise

with:

    network.language_model_patchwise_fused

before importing AutoRG_Brain.

This allows the GPT2 decoder to receive fused image features, while keeping
all original project files unchanged.

Important
---------
This script only switches the LanguageModel implementation to the fused version.
The actual multi-modal feature stacking/fusion depends on the upstream pipeline:

    single-modal / already fused feature: [B, P, D]
    multi-modal feature before fusion:   [B, M, P, D]

The fused language model will automatically convert:

    [B, M, P, D] -> mean over M -> [B, P, D]

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
# network.language_model_patchwise with the fused version.
#
# This avoids changing other import statements in the project.
# =========================================================

fused_lm = importlib.import_module("network.language_model_patchwise_fused")
sys.modules["network.language_model_patchwise"] = fused_lm

print("\n[FUSED INFERENCE] Module replacement finished.")
print("[FUSED INFERENCE] network.language_model_patchwise -> network.language_model_patchwise_fused")
print("[FUSED INFERENCE] Loaded module:", fused_lm.__name__)

from inference.inferenceSdk_fused import AutoRG_Brain
#from inference.inferenceSdk import AutoRG_Brain
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
        help="save image_hidden_states before GPT decoder, if supported by fused language model"
    )

    parser.add_argument(
        "--feature_dir",
        default=None,
        help="folder for saving intermediate image_hidden_states"
    )

    parser.add_argument(
        "--debug_fused",
        action="store_true",
        help="enable extra debug messages for fused-feature inference"
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
    # Environment variables for optional intermediate saving
    # =========================================================

    if args.save_intermediate:
        if args.feature_dir is None:
            args.feature_dir = str(out_dir / "features_fused")

        feature_dir = Path(args.feature_dir)
        feature_dir.mkdir(parents=True, exist_ok=True)

        os.environ["AUTORG_SAVE_INTERMEDIATE"] = "1"
        os.environ["AUTORG_FEATURE_DIR"] = str(feature_dir)

        print("\n[FUSED INFERENCE] save_intermediate enabled")
        print("[FUSED INFERENCE] feature_dir:", feature_dir.resolve())

    else:
        os.environ["AUTORG_SAVE_INTERMEDIATE"] = "0"
        print("\n[FUSED INFERENCE] save_intermediate disabled")

    if args.debug_fused:
        os.environ["AUTORG_DEBUG_FUSED"] = "1"
    else:
        os.environ["AUTORG_DEBUG_FUSED"] = "0"

    print("[FUSED INFERENCE] AUTORG_SAVE_INTERMEDIATE:", os.environ.get("AUTORG_SAVE_INTERMEDIATE"))
    print("[FUSED INFERENCE] AUTORG_FEATURE_DIR:", os.environ.get("AUTORG_FEATURE_DIR"))
    print("[FUSED INFERENCE] AUTORG_DEBUG_FUSED:", os.environ.get("AUTORG_DEBUG_FUSED"))

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

    print("\n[FUSED INFERENCE] Config:")
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

    print("\n[FUSED INFERENCE] Loaded test cases from:", test_file)
    print("[FUSED INFERENCE] Number of input items:", len(input_case_dict))

    # =========================================================
    # Run report generation
    # =========================================================

    results = model.report(input_case_dict)

    pred_path = out_dir / "pred_report_fused.json"

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n[FUSED INFERENCE] Finished inference.")
    print("[FUSED INFERENCE] pred_report saved to:", pred_path.resolve())

    if args.save_intermediate:
        print("[FUSED INFERENCE] features saved to:", Path(args.feature_dir).resolve())


if __name__ == "__main__":
    main()
