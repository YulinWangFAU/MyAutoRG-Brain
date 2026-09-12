#!/usr/bin/env python3
"""Paired statistical comparisons for case-level RaTEScore results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Method:
    category: str
    model: str
    input_source: str


@dataclass(frozen=True)
class Comparison:
    family: str
    label: str
    method_a: Method
    method_b: Method


DIRECT = Method("baseline", "Direct-AutoRG", "gt_abnormal_mask")
CONCAT = Method("feature_level", "Concat-Projection", "image_features")
TOKEN = Method("feature_level", "Token-Gated", "image_features")
MEAN = Method("feature_level", "Mean", "image_features")
WEIGHTED = Method("feature_level", "Weighted-Mean", "image_features")
T5_AUTO = Method("text_level", "T5-Small", "autorg_prediction")
VLM_FEW = Method("vlm", "Qwen2.5-VL-Few-Shot", "image_montage")
VLM_ZERO = Method("vlm", "Qwen2.5-VL-Zero-Shot", "image_montage")


def text_method(model: str, source: str) -> Method:
    return Method("text_level", model, source)


COMPARISONS = (
    Comparison("core", "Concat-Projection vs Direct AutoRG", CONCAT, DIRECT),
    Comparison("core", "Token-Gated vs Direct AutoRG", TOKEN, DIRECT),
    Comparison("core", "T5-Small vs Direct AutoRG", T5_AUTO, DIRECT),
    Comparison("core", "Concat-Projection vs T5-Small", CONCAT, T5_AUTO),
    Comparison("feature_ablation", "Concat-Projection vs Mean", CONCAT, MEAN),
    Comparison("feature_ablation", "Concat-Projection vs Weighted-Mean", CONCAT, WEIGHTED),
    Comparison("feature_ablation", "Concat-Projection vs Token-Gated", CONCAT, TOKEN),
    Comparison("feature_ablation", "Token-Gated vs Mean", TOKEN, MEAN),
    Comparison("vlm", "Qwen Few-Shot vs Zero-Shot", VLM_FEW, VLM_ZERO),
    Comparison("single_modal", "Direct AutoRG vs AutoRG-T1", DIRECT, Method("single_modal", "AutoRG-T1", "gt_abnormal_mask")),
    Comparison("single_modal", "Direct AutoRG vs AutoRG-T1C", DIRECT, Method("single_modal", "AutoRG-T1C", "gt_abnormal_mask")),
    Comparison("single_modal", "Direct AutoRG vs AutoRG-T2", DIRECT, Method("single_modal", "AutoRG-T2", "gt_abnormal_mask")),
    Comparison("single_modal", "Direct AutoRG vs AutoRG-FLAIR", DIRECT, Method("single_modal", "AutoRG-FLAIR", "gt_abnormal_mask")),
    Comparison(
        "oracle_input",
        "T5-Small GT input vs AutoRG input",
        text_method("T5-Small", "oracle_ground_truth"),
        text_method("T5-Small", "autorg_prediction"),
    ),
    Comparison(
        "oracle_input",
        "Flan-T5-LoRA GT input vs AutoRG input",
        text_method("Flan-T5-Large-LoRA", "oracle_ground_truth"),
        text_method("Flan-T5-Large-LoRA", "autorg_prediction"),
    ),
    Comparison(
        "oracle_input",
        "PullLoss GT input vs AutoRG input",
        text_method("Flan-T5-Large-LoRA-PullLoss", "oracle_ground_truth"),
        text_method("Flan-T5-Large-LoRA-PullLoss", "autorg_prediction"),
    ),
)


def method_key(method: Method) -> tuple[str, str, str]:
    return method.category, method.model, method.input_source


def read_scores(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    groups: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["category"], row["model"], row["input_source"])
            case_scores = groups.setdefault(key, {})
            case_id = row["case_id"]
            if case_id in case_scores:
                raise ValueError(f"Duplicate case {case_id!r} in group {key}")
            case_scores[case_id] = float(row["ratescore"])
    return groups


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def paired_bootstrap_ci(diff: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(diff), size=(iterations, len(diff)))
    return percentile_ci(diff[indices].mean(axis=1))


def paired_permutation_p(diff: np.ndarray, iterations: int, rng: np.random.Generator) -> float:
    observed = abs(float(diff.mean()))
    extreme = 0
    completed = 0
    batch_size = 10_000
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, len(diff)))
        permuted = np.abs((signs * diff).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
        completed += current
    return (extreme + 1) / (iterations + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def format_method(method: Method) -> str:
    return f"{method.category}/{method.model}/{method.input_source}"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "results_b1_gtmask" / "ratescore_per_case.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results_b1_gtmask" / "paired_statistics")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = read_scores(args.input)
    rows = []

    for index, comparison in enumerate(COMPARISONS):
        key_a = method_key(comparison.method_a)
        key_b = method_key(comparison.method_b)
        if key_a not in groups or key_b not in groups:
            raise KeyError(f"Missing comparison input: {key_a if key_a not in groups else key_b}")

        cases_a = groups[key_a]
        cases_b = groups[key_b]
        if set(cases_a) != set(cases_b):
            raise ValueError(f"Case mismatch for {comparison.label}")
        case_ids = sorted(cases_a)
        scores_a = np.array([cases_a[c] for c in case_ids], dtype=float)
        scores_b = np.array([cases_b[c] for c in case_ids], dtype=float)
        diff = scores_a - scores_b
        rng = np.random.default_rng(args.seed + index)
        ci_low, ci_high = paired_bootstrap_ci(diff, args.bootstrap_iterations, rng)
        p_value = paired_permutation_p(diff, args.permutation_iterations, rng)
        diff_std = float(diff.std(ddof=1))

        rows.append({
            "family": comparison.family,
            "comparison": comparison.label,
            "method_a": format_method(comparison.method_a),
            "method_b": format_method(comparison.method_b),
            "n": len(diff),
            "mean_a": float(scores_a.mean()),
            "mean_b": float(scores_b.mean()),
            "paired_mean_difference_a_minus_b": float(diff.mean()),
            "difference_ci95_low": ci_low,
            "difference_ci95_high": ci_high,
            "permutation_p_raw": p_value,
            "permutation_p_holm": 0.0,
            "paired_effect_size_dz": float(diff.mean() / diff_std) if diff_std else float("nan"),
            "a_wins": int(np.count_nonzero(diff > 0)),
            "ties": int(np.count_nonzero(diff == 0)),
            "b_wins": int(np.count_nonzero(diff < 0)),
        })

    adjusted = holm_adjust([row["permutation_p_raw"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["permutation_p_holm"] = adjusted_p

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ratescore_paired_comparisons.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.output_dir / "ratescore_paired_comparisons.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Paired RaTEScore comparisons\n\n")
        handle.write("Positive difference means method A scored higher than method B. ")
        handle.write("P-values are two-sided paired randomization tests with Holm correction across all comparisons.\n\n")
        handle.write("| Family | Comparison | Mean A | Mean B | A − B (95% CI) | Holm p | dz | A wins | B wins |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            diff_ci = (
                f"{row['paired_mean_difference_a_minus_b']:.4f} "
                f"[{row['difference_ci95_low']:.4f}, {row['difference_ci95_high']:.4f}]"
            )
            handle.write(
                f"| {row['family']} | {row['comparison']} | {row['mean_a']:.4f} | "
                f"{row['mean_b']:.4f} | {diff_ci} | {row['permutation_p_holm']:.4g} | "
                f"{row['paired_effect_size_dz']:.3f} | {row['a_wins']} | {row['b_wins']} |\n"
            )

    print(f"Validated and compared {len(rows)} paired hypotheses with N={rows[0]['n']} cases each")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
