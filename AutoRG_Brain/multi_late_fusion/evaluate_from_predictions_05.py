# -*- coding: utf-8 -*-
"""
Final Evaluation Script (Medical Report Generation)

Metrics:
- BLEU-2
- ROUGE-1
- METEOR
- BERTScore-F1
- RadGraph (Overall, Entity, Relation)
- RadCliQ (Composite, lower is better)

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
from radgraph import RadGraph

# =========================
# Prediction Files
# =========================

PREDICTION_FILES = {
    "T5-Small": "/home/woody/iwi5/iwi5325h/t5_late_fusion_model_20260218_151904/test_predictions_late_fusion_inputGTmodal.json",
    "Flan-T5-Large-LoRA": "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_20260218_234219/test_predictions_late_fusion_inputGTmodal.json",
    "Flan-T5-Large-LoRA-PullLoss": "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_PullLoss_20260228_164119/test_predictions_late_fusion_inputGTmodal.json"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================
# Load Metrics
# =========================

rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bleu_metric = BLEU(max_ngram_order=2)

# Initialize RadGraph
radgraph_model = RadGraph(device=DEVICE)

# =========================
# RadGraph Evaluation
# =========================

def compute_radgraph(predictions, references):

    results = radgraph_model(predictions, references)

    entity_f1_list = []
    relation_f1_list = []

    for r in results:
        entity_f1_list.append(r["entity_f1"])
        relation_f1_list.append(r["relation_f1"])

    entity_f1 = np.mean(entity_f1_list)
    relation_f1 = np.mean(relation_f1_list)
    overall_f1 = (entity_f1 + relation_f1) / 2

    return entity_f1, relation_f1, overall_f1


# =========================
# RadCliQ Evaluation
# =========================

def compute_radcliq(radgraph_f1, bert_f1, bleu2):

    # Normalize BLEU (0-100 → 0-1)
    bleu_norm = bleu2 / 100.0

    # Composite formula (论文近似权重版本)
    radcliq_score = 1 - (
        0.35 * radgraph_f1 +
        0.30 * bert_f1 +
        0.35 * bleu_norm
    )

    return radcliq_score


# =========================
# Main Evaluation Function
# =========================

def evaluate_model(pred_file):

    with open(pred_file) as f:
        data = json.load(f)

    predictions = [d["prediction"] for d in data]
    references = [d["reference"] for d in data]

    # ---------------------
    # BLEU-2
    # ---------------------
    bleu_score = bleu_metric.corpus_score(predictions, [references])
    bleu2 = bleu_score.score

    # ---------------------
    # ROUGE-1
    # ---------------------
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    # ---------------------
    # METEOR
    # ---------------------
    meteor_score = meteor.compute(
        predictions=predictions,
        references=references
    )["meteor"]

    # ---------------------
    # BERTScore
    # ---------------------
    P, R, F1 = bertscore(
        predictions,
        references,
        lang="en",
        device=DEVICE,
        verbose=False
    )
    bert_f1 = F1.mean().item()

    # ---------------------
    # RadGraph
    # ---------------------
    entity_f1, relation_f1, radgraph_f1 = compute_radgraph(predictions, references)

    # ---------------------
    # RadCliQ
    # ---------------------
    radcliq_score = compute_radcliq(
        radgraph_f1,
        bert_f1,
        bleu2
    )

    return {
        "BLEU-2 ↑": round(bleu2, 2),
        "ROUGE-1 ↑": round(rouge_result["rouge1"], 4),
        "METEOR ↑": round(meteor_score, 4),
        "BERTScore ↑": round(bert_f1, 4),
        "RadGraph ↑": round(radgraph_f1, 4),
        "RadGraph-Entity ↑": round(entity_f1, 4),
        "RadGraph-Relation ↑": round(relation_f1, 4),
        "RadCliQ ↓": round(radcliq_score, 4),
    }


# =========================
# Run Evaluation
# =========================

results = {}

for name, path in PREDICTION_FILES.items():
    print(f"\nEvaluating: {name}")
    metrics = evaluate_model(path)
    results[name] = metrics

df = pd.DataFrame(results).T

print("\n================ Final Comparison ================\n")
print(df.to_string())
print("\n=================================================\n")

# =========================
# Save Results
# =========================

OUTPUT_DIR = "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_PullLoss_20260228_164119"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results_final.csv"))

with open(os.path.join(OUTPUT_DIR, "evaluation_results_final.md"), "w") as f:
    f.write(df.to_markdown())

print("Results saved successfully.")