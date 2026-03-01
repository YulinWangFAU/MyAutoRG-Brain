
# -*- coding: utf-8 -*-
"""
Final Evaluation Script (Medical Report Generation - Research Grade)

Metrics:
- BLEU-2
- BLEU-4
- ROUGE-1
- METEOR
- BERTScore-F1
- RadGraph (micro: overall, entity, relation)
- RadCliQ (heuristic composite, lower better)
- Bootstrap 95% CI
- Paired Significance Test

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
from tqdm import tqdm

# =========================
# Config
# =========================

# =========================
# Prediction Files
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
# RadCliQ (Heuristic Version)
# =========================

def compute_radcliq(radgraph_f1, bert_f1, bleu2):
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

    # RadCliQ (heuristic)
    radcliq = compute_radcliq(radgraph_f1, bert_f1, bleu2)

    # Bootstrap CI for RadGraph
    per_sample_rad = rad_scores[1]
    ci_lower, ci_upper = bootstrap_ci(per_sample_rad, BOOTSTRAP_SAMPLES)

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
        "RadCliQ ↓": round(radcliq, 4),
        "PerSampleRad": per_sample_rad
    }

# =========================
# Run Evaluation
# =========================

results = {}

for name, path in PREDICTION_FILES.items():
    print(f"\nEvaluating: {name}")
    results[name] = evaluate_model(path)

# Paired Significance Test (RadGraph)
model_names = list(results.keys())
if len(model_names) == 2:
    m1, m2 = model_names
    t_stat, p_value = stats.ttest_rel(
        results[m1]["PerSampleRad"],
        results[m2]["PerSampleRad"]
    )
    print("\nPaired t-test on RadGraph:")
    print(f"{m1} vs {m2}: p = {p_value:.6f}")

# Remove per-sample before saving
for k in results:
    del results[k]["PerSampleRad"]

df = pd.DataFrame(results).T

print("\n================ Final Comparison ================\n")
print(df.to_string())
print("\n=================================================\n")

# Save
OUTPUT_DIR = "./evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"))
df.to_markdown(os.path.join(OUTPUT_DIR, "evaluation_results.md"))

print("Results saved successfully.")