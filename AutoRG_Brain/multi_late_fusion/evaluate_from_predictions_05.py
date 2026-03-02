# -*- coding: utf-8 -*-
"""
Final Evaluation Script (Publication-Ready Version)

Metrics:
- BLEU-2
- BLEU-4
- ROUGE-1
- METEOR
- BERTScore-F1
- RadGraph (micro: overall, entity, relation)
- Heuristic Composite Score
- Bootstrap 95% CI
- Wilcoxon Signed-Rank Test (Bonferroni corrected)

Author: Yulin Wang
"""

import json
import numpy as np
import evaluate
from sacrebleu.metrics import BLEU
from bert_score import score as bertscore
import torch
import os
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

print("Using device:", DEVICE)

# =========================
# Load Metrics
# =========================

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bleu2_metric = BLEU(max_ngram_order=2)
bleu4_metric = BLEU(max_ngram_order=4)

f1radgraph = F1RadGraph(
    reward_level="all",
    model_type="modern-radgraph-xl",
    device=DEVICE
)

# =========================
# Heuristic Composite Score
# =========================

def compute_composite(radgraph_f1, bert_f1, bleu2):
    bleu_norm = bleu2 / 100.0
    return 1 - (
        0.35 * radgraph_f1 +
        0.30 * bert_f1 +
        0.35 * bleu_norm
    )

# =========================
# Bootstrap CI
# =========================

def bootstrap_ci(metric_values, n_samples=1000):
    means = []
    n = len(metric_values)
    for _ in range(n_samples):
        sample = np.random.choice(metric_values, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    return lower, upper

# =========================
# Evaluate One Model
# =========================

def evaluate_model(pred_file):

    with open(pred_file) as f:
        data = json.load(f)

    predictions = [d["prediction"] for d in data]
    references = [d["reference"] for d in data]

    assert len(predictions) == len(references), "Prediction and reference length mismatch!"

    # BLEU
    bleu2 = bleu2_metric.corpus_score(predictions, [references]).score
    bleu4 = bleu4_metric.corpus_score(predictions, [references]).score

    # ROUGE
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    rouge1 = rouge_result["rouge1"]

    # METEOR
    meteor_score = meteor.compute(
        predictions=predictions,
        references=references
    )["meteor"]

    # BERTScore
    _, _, F1 = bertscore(
        predictions,
        references,
        lang="en",
        device=DEVICE,
        batch_size=BERT_BATCH_SIZE,
        verbose=False
    )
    bert_f1 = F1.mean().item()

    # RadGraph
    rad_scores = f1radgraph(
        hyps=predictions,
        refs=references
    )

    summary_tuple = rad_scores[0]
    reward_list = rad_scores[1]

    radgraph_f1 = float(summary_tuple[0])
    entity_f1 = float(summary_tuple[1])
    relation_f1 = float(summary_tuple[2])

    # =========================
    # 🔥 自动匹配 summary 与 per-sample index
    # =========================

    means = [np.mean([x[i] for x in reward_list]) for i in range(3)]
    idx = np.argmin([abs(means[i] - radgraph_f1) for i in range(3)])

    per_sample_rad = np.array([float(x[idx]) for x in reward_list])

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_ci(per_sample_rad, BOOTSTRAP_SAMPLES)

    composite = compute_composite(radgraph_f1, bert_f1, bleu2)

    return {
        "BLEU-2 ↑": round(bleu2, 2),
        "BLEU-4 ↑": round(bleu4, 2),
        "ROUGE-1 ↑": round(rouge1, 4),
        "METEOR ↑": round(meteor_score, 4),
        "BERTScore ↑": round(bert_f1, 4),
        "RadGraph ↑": round(radgraph_f1, 4),
        "RadGraph-Entity ↑": round(entity_f1, 4),
        "RadGraph-Relation ↑": round(relation_f1, 4),
        "RadGraph 95% CI": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "Composite ↓": round(composite, 4),
        "PerSampleRad": per_sample_rad
    }

# =========================
# Run Evaluation
# =========================

results = {}

for name, path in PREDICTION_FILES.items():
    print(f"\nEvaluating: {name}")
    results[name] = evaluate_model(path)

model_names = list(results.keys())

print("\n===== Wilcoxon Tests on RadGraph (per-sample) =====\n")

num_tests = len(list(combinations(model_names, 2)))
alpha = 0.05 / num_tests

print(f"Bonferroni-corrected alpha = {alpha:.6f}\n")

for m1, m2 in combinations(model_names, 2):

    scores1 = results[m1]["PerSampleRad"]
    scores2 = results[m2]["PerSampleRad"]

    try:
        stat, p_value = stats.wilcoxon(scores1, scores2)
    except ValueError:
        p_value = 1.0  # identical distributions

    print(f"{m1}  vs  {m2}")
    print(f"  p-value = {p_value:.6f}")

    if p_value < alpha:
        print(f"  Significant (Bonferroni corrected)")
    else:
        print("  n.s.")
    print()

# Remove per-sample before saving
for k in results:
    del results[k]["PerSampleRad"]

df = pd.DataFrame(results).T

print("\n================ Final Comparison ================\n")
print(df.to_string())
print("\n=================================================\n")

OUTPUT_DIR = "./evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"))
with open(os.path.join(OUTPUT_DIR, "evaluation_results.md"), "w") as f:
    f.write(df.to_markdown())

print("Results saved successfully.")