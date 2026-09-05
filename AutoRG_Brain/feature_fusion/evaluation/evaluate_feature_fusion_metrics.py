# -*- coding: utf-8 -*-
"""
Evaluate feature-level fusion report-generation outputs.

The metrics match the late-fusion evaluation scripts:
- BLEU-2
- BLEU-4
- ROUGE-1
- METEOR
- BERTScore-F1
- RadGraph overall/entity/relation F1
- Bootstrap 95% CI for RadGraph overall F1

Expected feature-fusion prediction format:
{
  "results": [
    {
      "case_id": "...",
      "pred_report": "...",
      "reference_report": "..."
    }
  ]
}

The script also accepts the late-fusion list format:
[
  {"prediction": "...", "reference": "..."}
]
"""

import argparse
import gc
import json
import os
from itertools import combinations
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from bert_score import score as bertscore
from radgraph import F1RadGraph
from sacrebleu.metrics import BLEU
from scipy import stats


DEFAULT_PRED_ROOT = Path("/home/woody/iwi5/iwi5325h/autorg_runs/feature_fusion_predictions")
DEFAULT_OUTPUT_DIR = Path("/home/woody/iwi5/iwi5325h/autorg_runs/feature_fusion_metric_results")

DEFAULT_EXPERIMENTS = {
    "FeatureFusion-Mean": "train_fusion_mean_layer2_b1_train322_val46",
    "FeatureFusion-WeightedMean": "train_fusion_weighted_mean_layer2_b1_train322_val46",
    "FeatureFusion-ConcatProjection": "train_fusion_concat_projection_layer2_b1_train322_val46",
    "FeatureFusion-TokenGated": "train_fusion_token_gated_layer2_b1_train322_val46",
}

EXPECTED_SPLIT_SIZES = {
    "training": 322,
    "validation": 46,
    "test": 92,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=["training", "validation", "test"],
        help="Splits to evaluate. Use all three for train/val/test tables.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bert_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--skip_radgraph", action="store_true")
    parser.add_argument(
        "--save_per_sample",
        action="store_true",
        help="Save per-sample predictions and RadGraph scores for later statistics.",
    )
    return parser.parse_args()


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bootstrap_ci(values, n_samples):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")

    means = []
    for _ in range(n_samples):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def load_prediction_records(path):
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported prediction JSON format: {path}")

    records = []
    for idx, row in enumerate(rows):
        pred = row.get("pred_report", row.get("prediction", ""))
        ref = row.get("reference_report", row.get("reference", row.get("target_text", "")))
        case_id = row.get("case_id", row.get("id", str(idx)))
        records.append(
            {
                "case_id": case_id,
                "prediction": str(pred).strip(),
                "reference": str(ref).strip(),
            }
        )

    return records


def get_predictions_and_refs(records):
    predictions = [r["prediction"] for r in records]
    references = [r["reference"] for r in records]
    return predictions, references


def exact_match_ratio(predictions, references):
    if not predictions:
        return float("nan")
    matches = [p.strip() == r.strip() for p, r in zip(predictions, references)]
    return float(np.mean(matches))


def compute_metrics(
    predictions,
    references,
    rouge,
    meteor,
    bleu2_metric,
    bleu4_metric,
    f1radgraph,
    device,
    bert_batch_size,
    bootstrap_samples,
    skip_bertscore=False,
    skip_radgraph=False,
):
    assert len(predictions) == len(references)

    bleu2 = bleu2_metric.corpus_score(predictions, [references]).score
    bleu4 = bleu4_metric.corpus_score(predictions, [references]).score

    rouge1 = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )["rouge1"]

    meteor_score = meteor.compute(
        predictions=predictions,
        references=references,
    )["meteor"]

    if skip_bertscore:
        bert_f1 = float("nan")
    else:
        try:
            _, _, f1 = bertscore(
                predictions,
                references,
                lang="en",
                device=device,
                batch_size=bert_batch_size,
                rescale_with_baseline=True,
                verbose=False,
            )
            bert_f1 = float(f1.mean().item())
            del f1
            clean_memory()
        except Exception as exc:
            print("BERTScore failed:", exc)
            bert_f1 = float("nan")

    per_sample_overall = np.array([float("nan")] * len(predictions))
    per_sample_entity = np.array([float("nan")] * len(predictions))
    per_sample_relation = np.array([float("nan")] * len(predictions))

    if skip_radgraph:
        overall_f1 = entity_f1 = relation_f1 = float("nan")
        ci_low = ci_high = float("nan")
    else:
        try:
            _, reward_list, *_ = f1radgraph(hyps=predictions, refs=references)
            reward_array = np.asarray(reward_list, dtype=float)

            if reward_array.ndim == 2:
                per_sample_overall = reward_array[:, 0]
                per_sample_entity = reward_array[:, 1]
                per_sample_relation = reward_array[:, 2]
            else:
                per_sample_overall = reward_array

            overall_f1 = float(np.nanmean(per_sample_overall))
            entity_f1 = float(np.nanmean(per_sample_entity))
            relation_f1 = float(np.nanmean(per_sample_relation))
            ci_low, ci_high = bootstrap_ci(per_sample_overall, bootstrap_samples)
        except Exception as exc:
            print("RadGraph failed:", exc)
            overall_f1 = entity_f1 = relation_f1 = float("nan")
            ci_low = ci_high = float("nan")

    return {
        "N": len(predictions),
        "ExactMatch": exact_match_ratio(predictions, references),
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "ROUGE-1": rouge1,
        "METEOR": meteor_score,
        "BERTScore": bert_f1,
        "RadGraph": overall_f1,
        "RadGraph-Entity": entity_f1,
        "RadGraph-Relation": relation_f1,
        "RadGraph-CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "PerSampleRadGraph": per_sample_overall,
        "PerSampleRadGraphEntity": per_sample_entity,
        "PerSampleRadGraphRelation": per_sample_relation,
    }


def run_wilcoxon(results_by_model, split, output_dir):
    rows = []
    model_names = list(results_by_model)
    pairs = list(combinations(model_names, 2))
    if not pairs:
        return pd.DataFrame()

    corrected_alpha = 0.05 / len(pairs)

    for model_a, model_b in pairs:
        a = results_by_model[model_a]["PerSampleRadGraph"]
        b = results_by_model[model_b]["PerSampleRadGraph"]
        mask = ~(np.isnan(a) | np.isnan(b))
        a = a[mask]
        b = b[mask]

        if len(a) < 10 or np.allclose(a, b):
            stat = float("nan")
            p_value = float("nan")
            effect_r = float("nan")
            significant = False
        else:
            stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
            if p_value > 0:
                z = stats.norm.ppf(p_value / 2)
                effect_r = abs(z) / np.sqrt(len(a))
            else:
                effect_r = 0.0
            significant = bool(p_value < corrected_alpha)

        rows.append(
            {
                "split": split,
                "model_a": model_a,
                "model_b": model_b,
                "n": len(a),
                "wilcoxon_stat": stat,
                "p_value": p_value,
                "bonferroni_alpha": corrected_alpha,
                "effect_size_r": effect_r,
                "significant": significant,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"feature_fusion_wilcoxon_radgraph_{split}.csv", index=False)
    return df


def rounded_display(metrics):
    out = {}
    for key, value in metrics.items():
        if key.startswith("PerSample"):
            continue
        if isinstance(value, float):
            if key.startswith("BLEU"):
                out[key] = round(value, 2)
            else:
                out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def save_per_sample_scores(path, records, metrics, model_name, split):
    rows = []
    overall = metrics["PerSampleRadGraph"]
    entity = metrics["PerSampleRadGraphEntity"]
    relation = metrics["PerSampleRadGraphRelation"]

    for idx, record in enumerate(records):
        rows.append(
            {
                "model": model_name,
                "split": split,
                "case_id": record["case_id"],
                "prediction": record["prediction"],
                "reference": record["reference"],
                "radgraph": float(overall[idx]) if idx < len(overall) else float("nan"),
                "radgraph_entity": float(entity[idx]) if idx < len(entity) else float("nan"),
                "radgraph_relation": float(relation[idx]) if idx < len(relation) else float("nan"),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Feature-level fusion metric evaluation")
    print("Device:", device)
    print("Prediction root:", args.pred_root)
    print("Output dir:", args.output_dir)
    print("Splits:", ", ".join(args.splits))
    print("=" * 80)

    print("Loading text metrics...")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bleu2_metric = BLEU(max_ngram_order=2)
    bleu4_metric = BLEU(max_ngram_order=4)

    if args.skip_radgraph:
        f1radgraph = None
    else:
        print("Loading RadGraph modern-radgraph-xl...")
        f1radgraph = F1RadGraph(
            reward_level="all",
            model_type="modern-radgraph-xl",
            device=device,
        )

    all_summary_rows = []

    for split in args.splits:
        print("\n" + "=" * 80)
        print("Evaluating split:", split)
        print("=" * 80)

        split_results = {}
        split_display = {}

        for model_name, exp_dir in DEFAULT_EXPERIMENTS.items():
            pred_file = args.pred_root / exp_dir / f"pred_{split}.json"
            if not pred_file.exists():
                print("Missing:", pred_file)
                continue

            records = load_prediction_records(pred_file)
            predictions, references = get_predictions_and_refs(records)
            expected_n = EXPECTED_SPLIT_SIZES.get(split)

            print(f"\nModel: {model_name}")
            print("File:", pred_file)
            print("Cases:", len(records), "Expected:", expected_n)
            if expected_n is not None and len(records) != expected_n:
                print("WARNING: unexpected number of cases")

            print("--- First sample sanity check ---")
            if records:
                print("case_id:", records[0]["case_id"])
                print("pred:", records[0]["prediction"][:500])
                print("ref :", records[0]["reference"][:500])
            print("---------------------------------")

            metrics = compute_metrics(
                predictions,
                references,
                rouge,
                meteor,
                bleu2_metric,
                bleu4_metric,
                f1radgraph,
                device,
                args.bert_batch_size,
                args.bootstrap_samples,
                skip_bertscore=args.skip_bertscore,
                skip_radgraph=args.skip_radgraph,
            )

            split_results[model_name] = metrics
            display_row = rounded_display(metrics)
            split_display[model_name] = display_row
            all_summary_rows.append({"split": split, "model": model_name, **display_row})

            if args.save_per_sample:
                safe_name = model_name.lower().replace("featurefusion-", "").replace("_", "-")
                save_per_sample_scores(
                    args.output_dir / f"feature_fusion_per_sample_{safe_name}_{split}.csv",
                    records,
                    metrics,
                    model_name,
                    split,
                )

        if not split_display:
            continue

        split_df = pd.DataFrame(split_display).T
        split_df.to_csv(args.output_dir / f"feature_fusion_metrics_{split}.csv")
        print("\nFinal comparison for", split)
        print(split_df.to_string())

        if not args.skip_radgraph:
            wilcoxon_df = run_wilcoxon(split_results, split, args.output_dir)
            if not wilcoxon_df.empty:
                print("\nWilcoxon tests on per-sample RadGraph for", split)
                print(wilcoxon_df.to_string(index=False))

    if all_summary_rows:
        all_df = pd.DataFrame(all_summary_rows)
        all_df.to_csv(args.output_dir / "feature_fusion_metrics_all_requested_splits.csv", index=False)
        print("\nSaved summary:", args.output_dir / "feature_fusion_metrics_all_requested_splits.csv")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
