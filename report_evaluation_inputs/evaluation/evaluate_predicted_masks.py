#!/usr/bin/env python3
"""Evaluate predicted AutoRG abnormal masks against binary BraTS GT masks."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk


MODALS = ("t1n", "t1c", "t2w", "t2f")
MODAL_LABELS = {"t1n": "T1", "t1c": "T1C", "t2w": "T2", "t2f": "FLAIR"}


def parse_prediction_name(path: Path) -> tuple[str, str]:
    suffix = "_ab.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected prediction filename: {path.name}")
    stem = path.name[: -len(suffix)]
    for modal in MODALS:
        marker = f"-{modal}"
        if stem.endswith(marker):
            return stem[: -len(marker)], modal
    raise ValueError(f"Cannot identify modality from {path.name}")


def assert_geometry(pred: sitk.Image, gt: sitk.Image, pred_path: Path, gt_path: Path) -> None:
    if pred.GetSize() != gt.GetSize():
        raise ValueError(f"Size mismatch: {pred_path} {pred.GetSize()} vs {gt_path} {gt.GetSize()}")
    checks = (
        ("spacing", pred.GetSpacing(), gt.GetSpacing(), 1e-5),
        ("origin", pred.GetOrigin(), gt.GetOrigin(), 1e-4),
        ("direction", pred.GetDirection(), gt.GetDirection(), 1e-5),
    )
    for name, left, right, atol in checks:
        if not np.allclose(left, right, rtol=0.0, atol=atol):
            raise ValueError(f"{name} mismatch: {pred_path} vs {gt_path}: {left} != {right}")


def safe_ratio(numerator: int, denominator: int, both_empty_value: float = 1.0) -> float:
    if denominator == 0:
        return both_empty_value
    return numerator / denominator


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = pred > 0
    gt = gt > 0
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    pred_voxels = int(np.count_nonzero(pred))
    gt_voxels = int(np.count_nonzero(gt))
    dice = safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = safe_ratio(tp, tp + fp + fn)
    precision = safe_ratio(tp, tp + fp, both_empty_value=1.0 if gt_voxels == 0 else 0.0)
    recall = safe_ratio(tp, tp + fn, both_empty_value=1.0)
    return {
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_empty": int(pred_voxels == 0),
        "gt_empty": int(gt_voxels == 0),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability))


def bootstrap_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize(rows: list[dict], group_name: str, group_value: str, iterations: int, seed: int) -> dict:
    result: dict[str, str | int | float] = {
        "group": group_name,
        "value": group_value,
        "n": len(rows),
        "empty_predictions": sum(int(row["pred_empty"]) for row in rows),
        "empty_prediction_rate": sum(int(row["pred_empty"]) for row in rows) / len(rows),
    }
    rng = np.random.default_rng(seed)
    for metric in ("dice", "iou", "precision", "recall"):
        values = np.array([float(row[metric]) for row in rows], dtype=float)
        low, high = bootstrap_ci(values, iterations, rng)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1))
        result[f"{metric}_median"] = float(np.median(values))
        result[f"{metric}_ci95_low"] = low
        result[f"{metric}_ci95_high"] = high
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/autorg_output_test_predmask"),
    )
    parser.add_argument(
        "--brats-test-root",
        type=Path,
        default=Path("/home/woody/iwi5/iwi5325h/BraTS_filtered/test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_a5_mask_metrics",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = sorted(args.prediction_dir.glob("*_ab.nii.gz"))
    if len(prediction_paths) != 368:
        raise ValueError(f"Expected 368 predicted masks, found {len(prediction_paths)}")

    rows = []
    seen = set()
    for index, pred_path in enumerate(prediction_paths, start=1):
        case_id, modal = parse_prediction_name(pred_path)
        key = (case_id, modal)
        if key in seen:
            raise ValueError(f"Duplicate prediction for {key}")
        seen.add(key)
        gt_path = args.brats_test_root / case_id / f"{case_id}-seg.nii.gz"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)

        pred_image = sitk.ReadImage(str(pred_path))
        gt_image = sitk.ReadImage(str(gt_path))
        assert_geometry(pred_image, gt_image, pred_path, gt_path)
        pred_array = sitk.GetArrayFromImage(pred_image)
        gt_array = sitk.GetArrayFromImage(gt_image)
        metrics = mask_metrics(pred_array, gt_array)
        rows.append({
            "case_id": case_id,
            "modal": modal,
            "modal_label": MODAL_LABELS[modal],
            "prediction_path": str(pred_path),
            "gt_path": str(gt_path),
            **metrics,
        })
        if index % 25 == 0 or index == len(prediction_paths):
            print(f"Processed {index}/{len(prediction_paths)} masks", flush=True)

    case_ids = {case_id for case_id, _ in seen}
    if len(case_ids) != 92:
        raise ValueError(f"Expected 92 cases, found {len(case_ids)}")
    for case_id in case_ids:
        present = {modal for found_case, modal in seen if found_case == case_id}
        if present != set(MODALS):
            raise ValueError(f"Incomplete modalities for {case_id}: {sorted(present)}")

    summary_rows = []
    for modal_index, modal in enumerate(MODALS):
        modal_rows = [row for row in rows if row["modal"] == modal]
        summary_rows.append(
            summarize(modal_rows, "modality", MODAL_LABELS[modal], args.bootstrap_iterations, args.seed + modal_index)
        )
    summary_rows.append(
        summarize(rows, "overall", "all_modality_inputs", args.bootstrap_iterations, args.seed + len(MODALS))
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mask_metrics_per_case.csv", rows)
    write_csv(args.output_dir / "mask_metrics_summary.csv", summary_rows)
    empty_rows = [row for row in rows if int(row["pred_empty"]) == 1]
    write_csv(args.output_dir / "empty_mask_failures.csv", empty_rows)

    print(f"Validated {len(case_ids)} cases and {len(rows)} modality-level masks")
    print(f"Empty predicted masks retained as failures: {len(empty_rows)}")
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
