#!/usr/bin/env python3
"""Seed- and case-aware paired statistics for text-fusion experiments."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


SEEDS = (42, 12345, 2026, 3407, 5901)
DEFAULT_MODELS = (
    "T5-Small",
    "Flan-T5-Large-LoRA",
    "Flan-T5-Large-LoRA-PullLoss",
)
SOURCES = ("gt_abnormal_mask", "predicted_abnormal_mask")
METRICS = (
    ("bleu4_pct", "BLEU-4", True),
    ("rouge1_recall_pct", "ROUGE-1", True),
    ("bertscore_f1_pct", "BERTScore", True),
    ("radgraph_er_f1_pct", "RadGraph", True),
    ("radcliq_v1", "RadCliQ-v1", False),
    ("ratescore_pct", "RaTEScore", True),
)
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            here
            / "results_text_fusion_multiseed_paper_metrics"
            / "paper_metrics_per_case.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            here
            / "results_text_fusion_multiseed_paper_metrics"
            / "paired_multiseed"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=50_000)
    parser.add_argument("--random-seed", type=int, default=20261026)
    parser.add_argument(
        "--models",
        nargs=3,
        default=list(DEFAULT_MODELS),
        metavar=("MODEL_1", "MODEL_2", "MODEL_3"),
        help="Exactly three model display names; all three pairs are tested.",
    )
    parser.add_argument("--analysis-name", default="Text-fusion")
    return parser.parse_args()


def load_data(
    path: Path, models: tuple[str, ...]
) -> tuple[dict[tuple[str, str, int, str], dict[str, float]], tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(models) * len(SOURCES) * len(SEEDS) * 92
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(rows)}")
    required = {"model", "input_source", "seed", "case_id"}
    required.update(metric for metric, _, _ in METRICS)
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data: dict[tuple[str, str, int, str], dict[str, float]] = {}
    cases_by_configuration: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        model = row["model"]
        source = row["input_source"]
        seed = int(row["seed"])
        case_id = row["case_id"]
        if model not in models or source not in SOURCES or seed not in SEEDS:
            raise ValueError(f"Unexpected configuration: {(model, source, seed)}")
        key = (model, source, seed, case_id)
        if key in data:
            raise ValueError(f"Duplicate row: {key}")
        data[key] = {metric: float(row[metric]) for metric, _, _ in METRICS}
        cases_by_configuration[(model, source, seed)].add(case_id)

    expected_configurations = {
        (model, source, seed)
        for model in models
        for source in SOURCES
        for seed in SEEDS
    }
    if set(cases_by_configuration) != expected_configurations:
        raise ValueError("Missing model/source/seed configurations")
    case_sets = list(cases_by_configuration.values())
    if any(len(cases) != 92 for cases in case_sets):
        raise ValueError("Every configuration must contain exactly 92 cases")
    if any(cases != case_sets[0] for cases in case_sets[1:]):
        raise ValueError("Case IDs do not align across configurations")
    return data, tuple(sorted(case_sets[0]))


def difference_matrix(
    data: dict[tuple[str, str, int, str], dict[str, float]],
    cases: tuple[str, ...],
    model_a: str,
    source_a: str,
    model_b: str,
    source_b: str,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                data[(model_a, source_a, seed, case_id)][metric]
                - data[(model_b, source_b, seed, case_id)][metric]
                for case_id in cases
            ]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )


def hierarchical_bootstrap_ci(
    differences: np.ndarray, iterations: int, rng: np.random.Generator
) -> tuple[float, float]:
    n_seeds, n_cases = differences.shape
    estimates: list[np.ndarray] = []
    batch_size = 500
    for start in range(0, iterations, batch_size):
        size = min(batch_size, iterations - start)
        sampled_seeds = rng.integers(0, n_seeds, size=(size, n_seeds))
        sampled_cases = rng.integers(0, n_cases, size=(size, n_cases))
        sampled = differences[
            sampled_seeds[:, :, None], sampled_cases[:, None, :]
        ]
        estimates.append(sampled.mean(axis=(1, 2)))
    values = np.concatenate(estimates)
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def clustered_randomization_p(
    differences: np.ndarray, iterations: int, rng: np.random.Generator
) -> float:
    # One independent cluster per held-out case; all five seed observations for
    # a case receive the same sign, preserving their within-case dependence.
    case_differences = differences.mean(axis=0)
    observed = abs(float(case_differences.mean()))
    extreme = 0
    batch_size = 2_000
    for start in range(0, iterations, batch_size):
        size = min(batch_size, iterations - start)
        signs = rng.choice((-1.0, 1.0), size=(size, len(case_differences)))
        permuted = (signs * case_differences).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed))
    return (extreme + 1.0) / (iterations + 1.0)


def holm_adjust(rows: list[dict[str, object]], p_field: str, output_field: str) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index][p_field]))
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(rows[index][p_field]))
        running = max(running, adjusted)
        rows[index][output_field] = f"{running:.8f}"


def analyze(
    data: dict[tuple[str, str, int, str], dict[str, float]],
    cases: tuple[str, ...],
    family: str,
    label: str,
    model_a: str,
    source_a: str,
    model_b: str,
    source_b: str,
    metric: str,
    metric_label: str,
    higher_is_better: bool,
    bootstrap_iterations: int,
    randomization_iterations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    differences = difference_matrix(
        data, cases, model_a, source_a, model_b, source_b, metric
    )
    average = float(differences.mean())
    ci_low, ci_high = hierarchical_bootstrap_ci(
        differences, bootstrap_iterations, rng
    )
    p_value = clustered_randomization_p(
        differences, randomization_iterations, rng
    )
    case_level = differences.mean(axis=0)
    case_sd = float(case_level.std(ddof=1))
    dz = average / case_sd if case_sd else float("nan")
    seed_means = differences.mean(axis=1)
    entity_a = model_a if source_a == source_b else source_a
    entity_b = model_b if source_a == source_b else source_b
    if higher_is_better:
        a_wins = int(np.count_nonzero(case_level > 0))
        b_wins = int(np.count_nonzero(case_level < 0))
        favored = entity_a if average > 0 else entity_b
    else:
        a_wins = int(np.count_nonzero(case_level < 0))
        b_wins = int(np.count_nonzero(case_level > 0))
        favored = entity_a if average < 0 else entity_b
    if average == 0:
        favored = "tie"
    return {
        "family": family,
        "comparison": label,
        "metric": metric,
        "metric_label": metric_label,
        "higher_is_better": higher_is_better,
        "model_a": model_a,
        "input_a": source_a,
        "model_b": model_b,
        "input_b": source_b,
        "n_seeds": len(SEEDS),
        "n_cases": len(cases),
        "mean_a_minus_b": f"{average:.8f}",
        "hierarchical_ci95_low": f"{ci_low:.8f}",
        "hierarchical_ci95_high": f"{ci_high:.8f}",
        "case_cluster_dz": f"{dz:.8f}",
        "raw_p": f"{p_value:.8f}",
        "favored": favored,
        "a_case_wins": a_wins,
        "b_case_wins": b_wins,
        "ties": len(cases) - a_wins - b_wins,
        **{
            f"difference_seed_{seed}": f"{value:.8f}"
            for seed, value in zip(SEEDS, seed_means)
        },
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1 or args.randomization_iterations < 1:
        raise ValueError("Iteration counts must be positive")
    models = tuple(args.models)
    if len(set(models)) != 3:
        raise ValueError("--models must contain three distinct model names")
    method_pairs = tuple(combinations(models, 2))
    data, cases = load_data(args.input, models)
    rng = np.random.default_rng(args.random_seed)
    rows: list[dict[str, object]] = []

    for source in SOURCES:
        for model_a, model_b in method_pairs:
            for metric, metric_label, higher_is_better in METRICS:
                rows.append(
                    analyze(
                        data,
                        cases,
                        "method_comparison",
                        f"{model_a} vs {model_b} ({source})",
                        model_a,
                        source,
                        model_b,
                        source,
                        metric,
                        metric_label,
                        higher_is_better,
                        args.bootstrap_iterations,
                        args.randomization_iterations,
                        rng,
                    )
                )

    for model in models:
        for metric, metric_label, higher_is_better in METRICS:
            rows.append(
                analyze(
                    data,
                    cases,
                    "mask_source",
                    f"Predicted vs GT mask ({model})",
                    model,
                    "predicted_abnormal_mask",
                    model,
                    "gt_abnormal_mask",
                    metric,
                    metric_label,
                    higher_is_better,
                    args.bootstrap_iterations,
                    args.randomization_iterations,
                    rng,
                )
            )

    for family in ("method_comparison", "mask_source"):
        family_rows = [row for row in rows if row["family"] == family]
        holm_adjust(family_rows, "raw_p", "holm_p_within_family")
    holm_adjust(rows, "raw_p", "holm_p_all_54")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paired_multiseed_comparisons.csv"
    md_path = args.output_dir / "paired_multiseed_comparisons.md"
    write_csv(csv_path, rows)

    lines = [
        f"# Paired multi-seed {args.analysis_name} comparisons",
        "",
        "Differences are A − B. Confidence intervals use a crossed hierarchical bootstrap over five seeds and 92 cases. Two-sided p-values use case-clustered paired randomization and are Holm-adjusted within the method-comparison and mask-source families.",
        "",
        "| Family | Comparison | Metric | A − B (95% CI) | Holm p | Favored |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        difference = float(row["mean_a_minus_b"])
        low = float(row["hierarchical_ci95_low"])
        high = float(row["hierarchical_ci95_high"])
        p_value = float(row["holm_p_within_family"])
        lines.append(
            f"| {row['family']} | {row['comparison']} | {row['metric_label']} | "
            f"{difference:.3f} [{low:.3f}, {high:.3f}] | {p_value:.4g} | "
            f"{row['favored']} |"
        )
    lines.extend(
        [
            "",
            "RadCliQ-v1 is lower-is-better; all other metrics are higher-is-better. The randomization test treats each held-out case as an independent cluster and keeps its five seed observations together.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Validated {len(models)} models x 2 input settings x "
        "5 seeds x 92 aligned cases"
    )
    print(f"Wrote {len(rows)} comparisons to {csv_path}")
    print(f"Wrote report to {md_path}")


if __name__ == "__main__":
    main()
