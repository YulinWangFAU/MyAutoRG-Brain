#!/usr/bin/env python3
"""Compute per-case and aggregate RaTEScore using RadEval."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from radeval import RadEval


GROUP_FIELDS = ("category", "model", "input_source")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_ci(values: list[float], iterations: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    boot_means = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(iterations))
    return percentile(boot_means, 0.025), percentile(boot_means, 0.975)


def parse_args() -> argparse.Namespace:
    evaluation_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=evaluation_dir / "prepared" / "unified_case_level_reports.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=evaluation_dir / "results")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in GROUP_FIELDS)].append(row)

    evaluator = RadEval(metrics=["ratescore"], per_sample=True)
    scored_rows = []
    summary_rows = []

    for group_index, (group_key, group_rows) in enumerate(sorted(groups.items())):
        group_rows.sort(key=lambda row: row["case_id"])
        refs = [row["reference"] for row in group_rows]
        hyps = [row["prediction"] for row in group_rows]
        scores = [float(value) for value in evaluator(refs=refs, hyps=hyps)["ratescore"]]

        if len(scores) != len(group_rows):
            raise RuntimeError(f"RaTEScore returned {len(scores)} scores for {len(group_rows)} reports")

        for row, score in zip(group_rows, scores):
            scored_rows.append({**row, "ratescore": f"{score:.8f}"})

        ci_low, ci_high = bootstrap_mean_ci(
            scores,
            iterations=args.bootstrap_iterations,
            seed=args.seed + group_index,
        )
        category, model, input_source = group_key
        summary_rows.append(
            {
                "category": category,
                "model": model,
                "input_source": input_source,
                "n": len(scores),
                "ratescore_mean": f"{mean(scores):.8f}",
                "ratescore_std": f"{stdev(scores):.8f}" if len(scores) > 1 else "0.00000000",
                "ratescore_median": f"{median(scores):.8f}",
                "ratescore_ci95_low": f"{ci_low:.8f}",
                "ratescore_ci95_high": f"{ci_high:.8f}",
            }
        )
        print(f"Scored {category} / {model} / {input_source}: N={len(scores)}")

    per_case_path = args.output_dir / "ratescore_per_case.csv"
    with per_case_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]) + ("ratescore",))
        writer.writeheader()
        writer.writerows(scored_rows)

    summary_path = args.output_dir / "ratescore_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote per-case scores to {per_case_path}")
    print(f"Wrote model summaries to {summary_path}")


if __name__ == "__main__":
    main()
