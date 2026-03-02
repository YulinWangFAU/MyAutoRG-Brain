# -*- coding: utf-8 -*-
"""
Final Evaluation Script (RadGraph Official-Compliant Version)

Metrics:
- BLEU-2
- BLEU-4
- ROUGE-1
- METEOR
- BERTScore-F1
- RadGraph (modern-radgraph-xl)
- Bootstrap 95% CI (RadGraph)
- Wilcoxon Signed-Rank Test (two-sided + Bonferroni)
- Effect Size (r)

Author: Yulin Wang
"""

import json
import numpy as np
import evaluate
from sacrebleu.metrics import BLEU
from bert_score import score as bertscore
import torch
import os
import gc
import pandas as pd
from radgraph import F1RadGraph
from scipy import stats
from itertools import combinations

# =========================
# Config
# =========================

PREDICTION_FILES = {
    "T5-Small": "/home/woody/iwi5/iwi5325h/t5_late_fusion_model_20260218_151904/test_predictions_late_fusion_inputGTmodal.json",
    "Flan-T5-Large-LoRA": "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_20260218_234219/test_predictions_late_fusion_inputGTmodal.json",
    "Flan-T5-Large-LoRA-PullLoss": "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_PullLoss_20260228_164119/test_predictions_late_fusion_inputGTmodal.json"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_BATCH_SIZE = 16
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("=" * 60)
print(f"Using device: {DEVICE}")
print("=" * 60)

# =========================
# Utility
# =========================

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def bootstrap_ci(values, n_samples=1000):
    values = np.array(values)
    means = []
    n = len(values)
    for _ in range(n_samples):
        sample = np.random.choice(values, size=n, replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

# =========================
# Load Metrics
# =========================

print("Loading metrics...")

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bleu2_metric = BLEU(max_ngram_order=2)
bleu4_metric = BLEU(max_ngram_order=4)

print("Loading RadGraph modern-radgraph-xl...")

f1radgraph = F1RadGraph(
    reward_level="all",
    model_type="modern-radgraph-xl",
    device=DEVICE
)

print("All metrics loaded.\n")

# =========================
# Evaluate Model
# =========================

def evaluate_model(pred_file):

    print(f"\nEvaluating: {os.path.basename(pred_file)}")

    with open(pred_file, "r") as f:
        data = json.load(f)

    predictions = [d["prediction"] for d in data]
    references = [d["reference"] for d in data]
    # ================= SANITY CHECK =================
    print("\n--- Sanity Check (first 3 samples) ---")
    for i in range(min(3, len(predictions))):
        print("Pred:", predictions[i])
        print("Ref :", references[i])
        print("-----")

    identical_count = sum(
        p.strip() == r.strip() for p, r in zip(predictions, references)
    )
    print("Exact identical ratio:",
          round(identical_count / len(predictions), 4))
    print("======================================\n")
    # =================================================
    assert len(predictions) == len(references)

    # ---------------- BLEU (correct format) ----------------
    references_bleu = [[ref] for ref in references]
    bleu2 = bleu2_metric.corpus_score(predictions, references_bleu).score
    bleu4 = bleu4_metric.corpus_score(predictions, references_bleu).score

    # ---------------- ROUGE ----------------
    rouge1 = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )["rouge1"]

    # ---------------- METEOR ----------------
    meteor_score = meteor.compute(
        predictions=predictions,
        references=references
    )["meteor"]

    # ---------------- BERTScore ----------------
    try:
        P, R, F1 = bertscore(
            predictions,
            references,
            lang="en",
            device=DEVICE,
            batch_size=BERT_BATCH_SIZE,
            rescale_with_baseline=True,
            verbose=False
        )
        bert_f1 = F1.mean().item()
        del P, R, F1
        clean_memory()
    except Exception as e:
        print("BERTScore failed:", e)
        bert_f1 = float("nan")

    # ---------------- RadGraph (Statistically Consistent Version) ----------------
    try:
        mean_scores, reward_list, *_ = f1radgraph(
            hyps=predictions,
            refs=references
        )

        reward_array = np.array(reward_list)

        if reward_array.ndim == 2:
            per_sample_overall = reward_array[:, 0]
            per_sample_entity = reward_array[:, 1]
            per_sample_relation = reward_array[:, 2]
        else:
            per_sample_overall = reward_array
            per_sample_entity = None
            per_sample_relation = None

        # 🔥 统一使用 sample mean（不要用 mean_scores）
        overall_f1 = float(np.mean(per_sample_overall))
        entity_f1 = float(np.mean(per_sample_entity)) if per_sample_entity is not None else float("nan")
        relation_f1 = float(np.mean(per_sample_relation)) if per_sample_relation is not None else float("nan")

        # Bootstrap CI 基于 same statistic
        ci_low, ci_high = bootstrap_ci(per_sample_overall, BOOTSTRAP_SAMPLES)

    except Exception as e:
        print("RadGraph failed:", e)
        overall_f1 = entity_f1 = relation_f1 = float("nan")
        per_sample_overall = np.array([float("nan")] * len(predictions))
        ci_low = ci_high = float("nan")

    return {
        "BLEU-2 ↑": round(bleu2, 2),
        "BLEU-4 ↑": round(bleu4, 2),
        "ROUGE-1 ↑": round(rouge1, 4),
        "METEOR ↑": round(meteor_score, 4),
        "BERTScore ↑": round(bert_f1, 4),
        "RadGraph ↑": round(overall_f1, 4),
        "RadGraph-Entity ↑": round(entity_f1, 4),
        "RadGraph-Relation ↑": round(relation_f1, 4),
        "RadGraph 95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "PerSampleRad": per_sample_overall
    }

# =========================
# Wilcoxon + Effect Size
# =========================

def run_wilcoxon(results, alpha=0.05):

    model_names = list(results.keys())
    pairs = list(combinations(model_names, 2))
    corrected_alpha = alpha / len(pairs)

    print("\nWilcoxon Signed-Rank Tests (two-sided)")
    print(f"Bonferroni corrected α = {corrected_alpha:.6f}\n")

    for m1, m2 in pairs:

        s1 = results[m1]["PerSampleRad"]
        s2 = results[m2]["PerSampleRad"]

        mask = ~(np.isnan(s1) | np.isnan(s2))
        s1, s2 = s1[mask], s2[mask]

        if len(s1) < 10:
            continue

        stat, p = stats.wilcoxon(s1, s2, alternative="two-sided")

        # Effect size r
        if p > 0:
            z = stats.norm.ppf(p / 2)
            r = abs(z) / np.sqrt(len(s1))
        else:
            r = 0.0

        print(f"{m1} vs {m2}")
        print(f"  p = {p:.6f}")
        print(f"  effect size r = {r:.4f}")
        print(f"  {'Significant' if p < corrected_alpha else 'Not significant'}\n")

# =========================
# Main
# =========================

def main():

    results = {}

    for name, path in PREDICTION_FILES.items():
        if os.path.exists(path):
            results[name] = evaluate_model(path)
        else:
            print(f"File not found: {path}")

    run_wilcoxon(results)

    display = {
        k: {kk: vv for kk, vv in v.items() if kk != "PerSampleRad"}
        for k, v in results.items()
    }

    df = pd.DataFrame(display).T

    print("\nFinal Comparison\n")
    print(df.to_string())

    os.makedirs("./evaluation_results", exist_ok=True)
    df.to_csv("./evaluation_results/evaluation_results.csv")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()