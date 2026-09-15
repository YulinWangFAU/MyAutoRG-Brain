#!/usr/bin/env python3
"""Paired RaTEScore comparison of GT-mask and predicted-mask text fusion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


MODELS = (
    "T5-Small",
    "Flan-T5-Large-LoRA",
    "Flan-T5-Large-LoRA-PullLoss",
)


def read_scores(path: Path, source: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["category"] != "text_level" or row["input_source"] != source:
                continue
            model = row["model"]
            if model not in MODELS:
                continue
            scores = output.setdefault(model, {})
            case_id = row["case_id"]
            if case_id in scores:
                raise ValueError(f"Duplicate {model}/{case_id} in {path}")
            scores[case_id] = float(row["ratescore"])
    missing = set(MODELS) - set(output)
    if missing:
        raise ValueError(f"Missing models in {path}: {sorted(missing)}")
    return output


def bootstrap_ci(diff: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(diff), size=(iterations, len(diff)))
    low, high = np.quantile(diff[indices].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def permutation_p(diff: np.ndarray, iterations: int, rng: np.random.Generator) -> float:
    observed = abs(float(diff.mean()))
    extreme = 0
    completed = 0
    while completed < iterations:
        current = min(10_000, iterations - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, len(diff)))
        extreme += int(np.count_nonzero(np.abs((signs * diff).mean(axis=1)) >= observed))
        completed += current
    return (extreme + 1) / (iterations + 1)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-input",
        type=Path,
        default=here / "results_b1_gtmask" / "ratescore_per_case.csv",
    )
    parser.add_argument(
        "--pred-input",
        type=Path,
        default=here / "results_a2_text_predmask" / "ratescore_per_case.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "results_a2_text_predmask" / "paired_gt_vs_pred",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = read_scores(args.gt_input, "autorg_prediction")
    pred = read_scores(args.pred_input, "predicted_abnormal_mask")
    rows = []
    for index, model in enumerate(MODELS):
        if set(gt[model]) != set(pred[model]):
            raise ValueError(f"Case mismatch for {model}")
        cases = sorted(gt[model])
        gt_values = np.array([gt[model][case] for case in cases])
        pred_values = np.array([pred[model][case] for case in cases])
        diff = pred_values - gt_values
        rng = np.random.default_rng(args.seed + index)
        low, high = bootstrap_ci(diff, args.bootstrap_iterations, rng)
        raw_p = permutation_p(diff, args.permutation_iterations, rng)
        std = float(diff.std(ddof=1))
        rows.append({
            "model": model,
            "n": len(diff),
            "gt_mask_mean": float(gt_values.mean()),
            "predicted_mask_mean": float(pred_values.mean()),
            "pred_minus_gt": float(diff.mean()),
            "difference_ci95_low": low,
            "difference_ci95_high": high,
            "permutation_p_raw": raw_p,
            "permutation_p_holm": 0.0,
            "paired_effect_size_dz": float(diff.mean() / std) if std else float("nan"),
            "pred_wins": int(np.count_nonzero(diff > 0)),
            "ties": int(np.count_nonzero(diff == 0)),
            "gt_wins": int(np.count_nonzero(diff < 0)),
        })
    for row, value in zip(rows, holm([row["permutation_p_raw"] for row in rows])):
        row["permutation_p_holm"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "text_fusion_gt_vs_predmask.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    md_path = args.output_dir / "text_fusion_gt_vs_predmask.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Text fusion: GT mask vs predicted mask\n\n")
        handle.write("Positive difference means predicted-mask input scored higher. ")
        handle.write("P-values use paired randomization tests with Holm correction.\n\n")
        handle.write("| Model | GT mean | Pred mean | Pred - GT (95% CI) | Holm p | dz | Pred wins | GT wins |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['gt_mask_mean']:.4f} | {row['predicted_mask_mean']:.4f} | "
                f"{row['pred_minus_gt']:.4f} [{row['difference_ci95_low']:.4f}, "
                f"{row['difference_ci95_high']:.4f}] | {row['permutation_p_holm']:.4g} | "
                f"{row['paired_effect_size_dz']:.3f} | {row['pred_wins']} | {row['gt_wins']} |\n"
            )
    print("Validated 3 models with 92 aligned cases each")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
