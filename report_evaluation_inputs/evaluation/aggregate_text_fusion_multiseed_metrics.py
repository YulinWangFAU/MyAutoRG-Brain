#!/usr/bin/env python3
"""Aggregate five-seed text-fusion metrics into paper-facing tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SEEDS = (42, 12345, 2026, 3407, 5901)
DEFAULT_MODELS = (
    "T5-Small",
    "Flan-T5-Large-LoRA",
    "Flan-T5-Large-LoRA-PullLoss",
)
INPUT_SOURCES = ("gt_abnormal_mask", "predicted_abnormal_mask")
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
            / "paper_metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "results_text_fusion_multiseed_paper_metrics" / "multiseed",
    )
    parser.add_argument(
        "--models",
        nargs=3,
        default=list(DEFAULT_MODELS),
        metavar=("MODEL_1", "MODEL_2", "MODEL_3"),
        help="Exactly three model display names in the desired table order.",
    )
    parser.add_argument("--analysis-name", default="Text-fusion")
    parser.add_argument("--output-prefix", default="text_fusion")
    parser.add_argument(
        "--category",
        default="text_level",
        choices=("text_level", "feature_level"),
    )
    return parser.parse_args()


def read_and_validate(
    path: Path, models: tuple[str, ...], category: str
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(models) * len(INPUT_SOURCES) * len(SEEDS)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} seed summaries, found {len(rows)}")

    required = {"category", "model", "input_source", "seed", "n"}
    required.update(f"{metric}_mean" for metric, _, _ in METRICS)
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (row["model"], row["input_source"], int(row["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate configuration: {key}")
        seen.add(key)
        if row["category"] != category:
            raise ValueError(f"Unexpected category for {key}: {row['category']}")
        if row["model"] not in models:
            raise ValueError(f"Unexpected model: {row['model']}")
        if row["input_source"] not in INPUT_SOURCES:
            raise ValueError(f"Unexpected input source: {row['input_source']}")
        if int(row["seed"]) not in SEEDS:
            raise ValueError(f"Unexpected seed: {row['seed']}")
        if int(row["n"]) != 92:
            raise ValueError(f"Expected n=92 for {key}, found {row['n']}")

    expected = {
        (model, source, seed)
        for model in models
        for source in INPUT_SOURCES
        for seed in SEEDS
    }
    if seen != expected:
        raise ValueError(f"Configuration mismatch; missing={sorted(expected - seen)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    args = parse_args()
    models = tuple(args.models)
    if len(set(models)) != 3:
        raise ValueError("--models must contain three distinct model names")
    if not args.output_prefix or any(char in args.output_prefix for char in "/\\"):
        raise ValueError("--output-prefix must be a non-empty filename prefix")
    rows = read_and_validate(args.input, models, args.category)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_configuration: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_seed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        by_configuration[(row["model"], row["input_source"])].append(row)
        by_seed[(row["model"], row["input_source"], int(row["seed"]))] = row

    aggregate_rows: list[dict[str, object]] = []
    for model in models:
        for source in INPUT_SOURCES:
            group = sorted(
                by_configuration[(model, source)], key=lambda row: int(row["seed"])
            )
            output: dict[str, object] = {
                "category": args.category,
                "model": model,
                "input_source": source,
                "n_seeds": len(group),
                "n_cases_per_seed": 92,
                "seeds": ";".join(str(seed) for seed in sorted(SEEDS)),
            }
            for metric, _, _ in METRICS:
                values = [float(row[f"{metric}_mean"]) for row in group]
                output[f"{metric}_mean_across_seeds"] = f"{mean(values):.8f}"
                output[f"{metric}_sd_across_seeds"] = f"{stdev(values):.8f}"
            aggregate_rows.append(output)

    per_seed_difference_rows: list[dict[str, object]] = []
    difference_summary_rows: list[dict[str, object]] = []
    for model in models:
        for metric, label, higher_is_better in METRICS:
            differences: list[float] = []
            for seed in SEEDS:
                gt = float(
                    by_seed[(model, "gt_abnormal_mask", seed)][f"{metric}_mean"]
                )
                pred = float(
                    by_seed[(model, "predicted_abnormal_mask", seed)][
                        f"{metric}_mean"
                    ]
                )
                difference = pred - gt
                differences.append(difference)
                per_seed_difference_rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "metric_label": label,
                        "higher_is_better": higher_is_better,
                        "seed": seed,
                        "gt_mask_mean": f"{gt:.8f}",
                        "predicted_mask_mean": f"{pred:.8f}",
                        "predicted_minus_gt": f"{difference:.8f}",
                    }
                )
            difference_summary_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "metric_label": label,
                    "higher_is_better": higher_is_better,
                    "n_seeds": len(differences),
                    "predicted_minus_gt_mean": f"{mean(differences):.8f}",
                    "predicted_minus_gt_sd": f"{stdev(differences):.8f}",
                }
            )

    aggregate_path = args.output_dir / f"{args.output_prefix}_multiseed_summary.csv"
    per_seed_path = args.output_dir / "mask_effect_per_seed.csv"
    difference_path = args.output_dir / "mask_effect_summary.csv"
    markdown_path = args.output_dir / f"{args.output_prefix}_multiseed_summary.md"
    write_csv(aggregate_path, aggregate_rows)
    write_csv(per_seed_path, per_seed_difference_rows)
    write_csv(difference_path, difference_summary_rows)

    aggregate_lookup = {
        (str(row["model"]), str(row["input_source"])): row
        for row in aggregate_rows
    }
    lines = [
        f"# {args.analysis_name} results across five random seeds",
        "",
        "Values are mean ± sample SD across seeds 42, 12345, 2026, 3407, and 5901. Each seed was evaluated on the same 92 held-out cases.",
        "",
        "| Model | Input | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for model in models:
        for source in INPUT_SOURCES:
            row = aggregate_lookup[(model, source)]
            values = []
            for metric, _, _ in METRICS:
                metric_mean = float(row[f"{metric}_mean_across_seeds"])
                metric_sd = float(row[f"{metric}_sd_across_seeds"])
                values.append(f"{fmt(metric_mean)} ± {fmt(metric_sd)}")
            source_label = "GT abnormal mask" if source == "gt_abnormal_mask" else "Predicted abnormal mask"
            lines.append(f"| {model} | {source_label} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "RadCliQ-v1 is lower-is-better; all other metrics are higher-is-better. Metric means are first computed over the 92 cases within each seed, then summarized across the five seeds.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        f"Validated {len(models)} models x 2 input settings x "
        "5 seeds x 92 cases"
    )
    print(f"Wrote {len(aggregate_rows)} aggregate rows to {aggregate_path}")
    print(f"Wrote {len(per_seed_difference_rows)} paired seed rows to {per_seed_path}")
    print(f"Wrote {len(difference_summary_rows)} mask-effect rows to {difference_path}")
    print(f"Wrote paper-ready table to {markdown_path}")


if __name__ == "__main__":
    main()
