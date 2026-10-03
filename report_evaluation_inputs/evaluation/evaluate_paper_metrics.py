#!/usr/bin/env python3
"""Evaluate reports with the metric definitions used by AutoRG-Brain.

The paper-facing columns use the same display scale as AutoRG-Brain:
BLEU-4, ROUGE-1, BERTScore, RadGraph and RaTEScore are percentages;
RadCliQ-v1 is the raw predicted error score (lower is better).
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import random
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev


GROUP_FIELDS = ("category", "model", "input_source")
OPTIONAL_GROUP_FIELDS = ("seed", "run_seed")
TOKEN_PATTERN = re.compile(r"\S+")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "prepared" / "unified_case_level_reports.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "results_paper_metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42, help="bootstrap seed only")
    parser.add_argument(
        "--radgraph-model",
        default="modern-radgraph-xl",
        choices=("radgraph", "radgraph-xl", "modern-radgraph-xl", "echograph"),
    )
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-radgraph", action="store_true")
    parser.add_argument("--skip-radcliq", action="store_true")
    parser.add_argument("--skip-ratescore", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {*GROUP_FIELDS, "case_id", "prediction", "reference"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return rows


def tokens(text: str) -> list[str]:
    # The released AutoRG code lowercases and whitespace-tokenizes reports.
    return TOKEN_PATTERN.findall(text.lower().strip())


def autorg_bleu4(reference: str, prediction: str) -> float:
    """Pure sentence-level 4-gram BLEU used in the AutoRG-Brain paper."""
    from nltk.translate.bleu_score import sentence_bleu

    ref_tokens = tokens(reference)
    hyp_tokens = tokens(prediction)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(
            sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                weights=(0.0, 0.0, 0.0, 1.0),
            )
        )


def rouge1_recall(reference: str, prediction: str) -> float:
    """Clipped unigram recall following the formula in AutoRG-Brain."""
    ref_counts = Counter(tokens(reference))
    hyp_counts = Counter(tokens(prediction))
    denominator = sum(ref_counts.values())
    if denominator == 0:
        return 1.0 if not hyp_counts else 0.0
    overlap = sum(min(count, hyp_counts[token]) for token, count in ref_counts.items())
    return overlap / denominator


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_ci(
    values: list[float], iterations: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    boot_means = sorted(
        mean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(iterations)
    )
    return percentile(boot_means, 0.025), percentile(boot_means, 0.975)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def compute_bertscore(
    references: list[str], predictions: list[str], device: str, batch_size: int
) -> list[float]:
    try:
        # RadEval 2.2+ vendors a transformers-5-compatible BERTScore.
        from radeval.metrics.bertscore._vendor import BERTScorer
    except ImportError:
        from bert_score import BERTScorer

    scorer = BERTScorer(
        lang="en",
        model_type="bert-base-uncased",
        num_layers=9,
        rescale_with_baseline=False,
        device=device,
        batch_size=batch_size,
    )
    _, _, f1 = scorer.score(predictions, references, batch_size=batch_size)
    return [float(value) for value in f1.cpu().tolist()]


def normalize_radgraph_rewards(reward_list: object, count: int) -> list[float]:
    """Return per-case RG_ER regardless of radgraph package array orientation."""
    import numpy as np

    rewards = np.asarray(reward_list, dtype=float)
    if rewards.shape == (3, count):
        return rewards[1, :].tolist()
    if rewards.shape == (count, 3):
        return rewards[:, 1].tolist()
    raise RuntimeError(
        "Unexpected F1RadGraph reward shape "
        f"{rewards.shape}; expected (3, {count}) or ({count}, 3)"
    )


def compute_radgraph(
    references: list[str], predictions: list[str], device: str, model_type: str
) -> list[float]:
    try:
        # Prefer RadEval's patched RadGraph, compatible with transformers 5.x.
        from radeval.metrics.radgraph import F1RadGraph
    except ImportError:
        from radgraph import F1RadGraph

    scorer = F1RadGraph(reward_level="all", model_type=model_type, device=device)
    _, reward_list, *_ = scorer(hyps=predictions, refs=references)
    return normalize_radgraph_rewards(reward_list, len(references))


def compute_radeval_metric(
    metric_name: str,
    output_key: str,
    references: list[str],
    predictions: list[str],
) -> list[float]:
    from radeval import RadEval

    result = RadEval(metrics=[metric_name], per_sample=True)(
        refs=references,
        hyps=predictions,
    )
    if output_key not in result:
        raise RuntimeError(
            f"RadEval metric {metric_name!r} returned {sorted(result)}; "
            f"expected {output_key!r}. Check the installed RadEval version."
        )
    values = [float(value) for value in result[output_key]]
    if len(values) != len(references):
        raise RuntimeError(
            f"{metric_name} returned {len(values)} values for {len(references)} reports"
        )
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    group_fields = GROUP_FIELDS + tuple(
        field for field in OPTIONAL_GROUP_FIELDS if field in rows[0]
    )
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        groups[tuple(row[field] for field in group_fields)].append(row_index)

    summary_rows: list[dict[str, object]] = []
    enabled_metrics = ["bleu4_pct", "rouge1_recall_pct"]
    if not args.skip_bertscore:
        enabled_metrics.append("bertscore_f1_pct")
    if not args.skip_radgraph:
        enabled_metrics.append("radgraph_er_f1_pct")
    if not args.skip_radcliq:
        enabled_metrics.append("radcliq_v1")
    if not args.skip_ratescore:
        enabled_metrics.append("ratescore_pct")

    references = [row["reference"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    scores: dict[str, list[float]] = {
        "bleu4_pct": [
            100.0 * autorg_bleu4(ref, hyp)
            for ref, hyp in zip(references, predictions)
        ],
        "rouge1_recall_pct": [
            100.0 * rouge1_recall(ref, hyp)
            for ref, hyp in zip(references, predictions)
        ],
    }
    if not args.skip_bertscore:
        scores["bertscore_f1_pct"] = [
            100.0 * value
            for value in compute_bertscore(
                references, predictions, args.device, args.batch_size
            )
        ]
    if not args.skip_radgraph:
        scores["radgraph_er_f1_pct"] = [
            100.0 * value
            for value in compute_radgraph(
                references, predictions, args.device, args.radgraph_model
            )
        ]
    if not args.skip_radcliq:
        # RadEval's per-sample values are the original raw RadCliQ-v1
        # predicted-error scores. Its aggregate value is inverted for
        # leaderboard use, so we intentionally ignore that aggregate.
        scores["radcliq_v1"] = compute_radeval_metric(
            "radcliq", "radcliq_v1", references, predictions
        )
    if not args.skip_ratescore:
        scores["ratescore_pct"] = [
            100.0 * value
            for value in compute_radeval_metric(
                "ratescore", "ratescore", references, predictions
            )
        ]

    per_case_rows = [
        {
            **row,
            **{metric: f"{scores[metric][row_index]:.8f}" for metric in enabled_metrics},
        }
        for row_index, row in enumerate(rows)
    ]

    for group_index, (group_key, group_indices) in enumerate(sorted(groups.items())):
        group_indices.sort(key=lambda index: rows[index]["case_id"])

        summary: dict[str, object] = {
            **dict(zip(group_fields, group_key)),
            "n": len(group_indices),
        }
        for metric_index, metric in enumerate(enabled_metrics):
            values = [scores[metric][index] for index in group_indices]
            ci_low, ci_high = bootstrap_mean_ci(
                values,
                args.bootstrap_iterations,
                args.seed + group_index * len(enabled_metrics) + metric_index,
            )
            summary.update(
                {
                    f"{metric}_mean": f"{mean(values):.8f}",
                    f"{metric}_case_std": f"{stdev(values):.8f}" if len(values) > 1 else "0.00000000",
                    f"{metric}_median": f"{median(values):.8f}",
                    f"{metric}_ci95_low": f"{ci_low:.8f}",
                    f"{metric}_ci95_high": f"{ci_high:.8f}",
                }
            )
        summary_rows.append(summary)
        print(f"Scored {' / '.join(group_key)}: N={len(group_indices)}")

    write_csv(args.output_dir / "paper_metrics_per_case.csv", per_case_rows)
    write_csv(args.output_dir / "paper_metrics_summary.csv", summary_rows)

    methodology = {
        "input": str(args.input.resolve()),
        "group_fields": group_fields,
        "statistics": {
            "case_std": "sample standard deviation across test cases",
            "ci95": "nonparametric bootstrap confidence interval for the case mean",
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.seed,
        },
        "metrics": {
            "bleu4_pct": {
                "implementation": "nltk sentence_bleu",
                "weights": [0, 0, 0, 1],
                "tokenization": "lowercase whitespace tokens",
                "smoothing": "none",
                "scale": 100,
                "direction": "higher",
            },
            "rouge1_recall_pct": {
                "implementation": "clipped unigram recall",
                "tokenization": "lowercase whitespace tokens",
                "scale": 100,
                "direction": "higher",
            },
            "bertscore_f1_pct": {
                "model_type": "bert-base-uncased",
                "rescale_with_baseline": False,
                "scale": 100,
                "direction": "higher",
                "enabled": not args.skip_bertscore,
            },
            "radgraph_er_f1_pct": {
                "implementation": "radgraph.F1RadGraph RG_ER",
                "model_type": args.radgraph_model,
                "reward_level": "all",
                "scale": 100,
                "direction": "higher",
                "enabled": not args.skip_radgraph,
            },
            "radcliq_v1": {
                "implementation": "RadEval RadCliQ-v1 per-sample raw predicted-error score",
                "scale": 1,
                "direction": "lower",
                "domain_warning": "RadCliQ-v1 was trained for chest X-ray reports",
                "enabled": not args.skip_radcliq,
            },
            "ratescore_pct": {
                "implementation": "RadEval RaTEScore",
                "scale": 100,
                "direction": "higher",
                "enabled": not args.skip_ratescore,
            },
        },
        "package_versions": {
            name: package_version(name)
            for name in ("nltk", "bert-score", "radgraph", "radeval", "torch", "transformers")
        },
    }
    (args.output_dir / "paper_metrics_methodology.json").write_text(
        json.dumps(methodology, indent=2), encoding="utf-8"
    )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
