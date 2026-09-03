import argparse
import importlib
import json
import os
import sys
import types
from collections import OrderedDict

import numpy as np
import torch
from torch.cuda.amp import autocast


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

nnunet_alias = types.ModuleType("nnunet")
nnunet_alias.__path__ = [os.path.dirname(os.path.abspath(__file__))]
sys.modules.setdefault("nnunet", nnunet_alias)
sys.modules.setdefault("nnunet.paths", importlib.import_module("paths"))

fused_lm = importlib.import_module("network.language_model_patchwise_fused")
sys.modules["network.language_model_patchwise"] = fused_lm

feature_trainer = importlib.import_module("network_training.nnUNetTrainerV2_llm_feature_fusion")
sys.modules["network_training.nnUNetTrainerV2_llm_resize_multi"] = feature_trainer

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from paths import default_plans_identifier
from run.default_configuration import get_default_configuration
from dataset.dataset_loading_llm_multi import DataLoader3D_Multi as DataLoader3D
from utilities.task_name_id_conversion import convert_id_to_task_name
from utilities.to_torch import maybe_to_torch, to_cuda


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate feature-level fusion reports from a trained checkpoint."
    )
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="Task id/name, for example 003 or Task003_llm_fusion.")
    parser.add_argument("fold", help="Fold id, for example 0.")

    parser.add_argument("-train", "--train_file", required=True, help="Feature-fusion train/test json.")
    parser.add_argument("--plans_file", required=True, help="The segmentation nnU-Net plans pkl.")
    parser.add_argument("--checkpoint", required=True, help="Path to model_final_checkpoint.model or model_best*.model.")

    parser.add_argument("--dataset", default="six")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--feature_layer", type=int, default=2)
    parser.add_argument("--network_type", default="share")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--input_mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--single_modal_index", type=int, default=0)

    parser.add_argument(
        "--fusion_type",
        choices=["mean", "weighted_mean", "token_gated", "concat_projection"],
        required=True,
        help="Must match the fusion_type used during training.",
    )
    parser.add_argument("--fusion_gate_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)

    parser.add_argument(
        "--split",
        choices=["training", "validation", "test"],
        default="validation",
        help="Which split in the train json to generate reports for.",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        help="Defaults to the checkpoint folder.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Defaults to pred_report_feature_level_{fusion_type}_{split}.json.",
    )
    parser.add_argument("-p", default=default_plans_identifier, required=False)
    parser.add_argument("--fp32", action="store_true", default=False)
    return parser.parse_args()


def flatten_report(report):
    return [phrase for case_report in report for phrase in case_report]


def get_loader_for_split(trainer, split):
    if split == "training":
        return trainer.dl_tr
    if split == "validation":
        return trainer.dl_val

    test_cases = trainer.train_file.get("test")
    if test_cases is None:
        raise KeyError(
            "split='test' was requested, but the train json has no top-level 'test' key."
        )

    missing_cases = [case_id for case_id in test_cases if case_id not in trainer.dataset]
    if missing_cases:
        raise KeyError(
            "Some test cases are not present in the preprocessed dataset folder: "
            + ", ".join(missing_cases[:10])
        )

    test_dataset = OrderedDict((case_id, trainer.dataset[case_id]) for case_id in test_cases)
    test_report = trainer.train_file.get("region_report", {}).get("test")
    if test_report is None:
        test_report = {case_id: {"": "abnormal"} for case_id in test_cases}

    return DataLoader3D(
        test_dataset,
        trainer.patch_size,
        trainer.patch_size,
        trainer.batch_size,
        report=test_report,
        has_prev_stage=True,
        oversample_foreground_percent=trainer.oversample_foreground_percent,
        pad_mode="constant",
        pad_sides=trainer.pad_all_sides,
        memmap_mode="r",
        input_mode=trainer.input_mode,
        single_modal_index=trainer.single_modal_index,
    )


def generate_one_case(trainer, data_loader, case_id, mixed_precision=True):
    old_keys = list(data_loader.list_of_keys)
    data_loader.list_of_keys = [case_id]

    try:
        data_dict = data_loader.generate_train_batch()
    finally:
        data_loader.list_of_keys = old_keys

    data = maybe_to_torch(data_dict["data"])
    target = maybe_to_torch(data_dict.get("target", data_dict.get("seg")))
    modal = data_dict["modal"]
    region, reference_report_nested = trainer.split_batch_report(data_dict["report"])
    reference_reports = flatten_report(reference_report_nested)

    if torch.cuda.is_available():
        data = to_cuda(data)
        target = to_cuda(target)

    target_device = trainer.llm_model.module.device if torch.cuda.device_count() > 1 else trainer.llm_model.device

    with torch.no_grad():
        with autocast(enabled=mixed_precision):
            region_features, _ = trainer.network(
                data,
                target,
                modal,
                region,
                eval_mode="region_oracle",
            )

            if isinstance(region_features, (list, tuple)):
                region_features = torch.stack(
                    [item.detach().to(target_device) for item in region_features],
                    dim=0,
                )
            else:
                region_features = region_features.detach().to(target_device)

            output = trainer.llm_model.generate(
                region_features,
                max_length=300,
                num_beams=1,
                num_beam_groups=1,
                do_sample=False,
                num_return_sequences=1,
                early_stopping=True,
            )

    generated_reports = trainer.tokenizer.batch_decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    generated_reports = [item.strip() for item in generated_reports]

    return {
        "case_id": case_id,
        "fusion_type": trainer.llm_model.fusion_type,
        "pred_reports": generated_reports,
        "pred_report": " ".join([item for item in generated_reports if item]),
        "reference_reports": reference_reports,
        "reference_report": " ".join([item for item in reference_reports if item]),
    }


def main():
    args = parse_args()

    # Report inference is deterministic and should process one case at a time.
    os.environ.setdefault("AUTORG_REAL_BATCH_SIZE", "1")
    os.environ.setdefault("AUTORG_AUG_NUM_THREADS", "1")
    os.environ.setdefault("AUTORG_AUG_CACHE_PER_THREAD", "1")

    os.environ["AUTORG_FUSION_TYPE"] = args.fusion_type
    os.environ["AUTORG_FUSION_GATE_HIDDEN_DIM"] = str(args.fusion_gate_hidden_dim)
    os.environ["AUTORG_FUSION_DROPOUT"] = str(args.fusion_dropout)
    os.environ["AUTORG_NUM_MODALITIES"] = "4"

    task = args.task
    if not task.startswith("Task"):
        task = convert_id_to_task_name(int(task))

    fold = args.fold if args.fold == "all" else int(args.fold)
    run_mixed_precision = not args.fp32

    plans_file, output_folder_name, dataset_directory, batch_dice, stage = get_default_configuration(
        args.network,
        task,
        args.network_trainer,
        args.p,
        plans_file=args.plans_file,
    )

    trainer_cls = feature_trainer.nnUNetTrainerV2
    trainer = trainer_cls(
        plans_file,
        fold,
        args.train_file,
        only_ana=False,
        abnormal_type="intense",
        num_batches_per_epoch=1,
        num_val_batches_per_epoch=1,
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=True,
        deterministic=True,
        fp16=run_mixed_precision,
        network_type=args.network_type,
        feature_layer=args.feature_layer,
        dataset_directory_bucket=None,
        train_with_seg=False,
        avg_type="xyz",
        pool_to_feature_layer=None,
        use_conv_pool=False,
        use_global=False,
        size=args.size,
        dataset=args.dataset,
        max_tokens=args.max_tokens,
        input_mode=args.input_mode,
        single_modal_index=args.single_modal_index,
    )

    trainer.initialize(training=True, no_aug=True)
    trainer.load_checkpoint(args.checkpoint, train=False)
    trainer.network.eval()
    trainer.llm_model.eval()

    split_cases = trainer.train_file.get(args.split)
    if split_cases is None:
        raise KeyError(
            f"Split '{args.split}' is not present in {args.train_file}. "
            f"Available keys: {list(trainer.train_file.keys())}"
        )

    data_loader = get_loader_for_split(trainer, args.split)
    output_records = []

    for idx, case_id in enumerate(split_cases):
        print(f"[FEATURE FUSION INFER] {idx + 1}/{len(split_cases)} {case_id}")
        output_records.append(
            generate_one_case(trainer, data_loader, case_id, mixed_precision=run_mixed_precision)
        )

    output_folder = args.output_folder or os.path.dirname(os.path.abspath(args.checkpoint))
    maybe_mkdir_p(output_folder)
    output_name = args.output_name or f"pred_report_feature_level_{args.fusion_type}_{args.split}.json"
    output_path = join(output_folder, output_name)

    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "fusion_type": args.fusion_type,
        "split": args.split,
        "task": task,
        "fold": fold,
        "num_cases": len(output_records),
        "results": output_records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[FEATURE FUSION INFER] Saved:", output_path)


if __name__ == "__main__":
    main()
