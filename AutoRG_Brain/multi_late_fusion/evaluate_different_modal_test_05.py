# -*- coding: utf-8 -*-
"""
Created on 2026/2/19 13:16

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Evaluate prediction files:
BLEU-4, ROUGE-1, ROUGE-L, BERTScore-F1

Author: Yulin Wang
"""

import json
import numpy as np
import evaluate
from sacrebleu import corpus_bleu
from bert_score import score as bertscore
import torch
import os
# -*- coding: utf-8 -*-
"""
Evaluate prediction files and print formatted comparison table

BLEU-4, ROUGE-1, ROUGE-L, BERTScore-F1
"""

import json
import numpy as np
import evaluate
from sacrebleu import corpus_bleu
from bert_score import score as bertscore
import torch
import pandas as pd

# =========================
# 修改这里：两个 prediction 文件路径
# =========================

PREDICTION_FILES = {
    "T5-Small": "/home/woody/iwi5/iwi5325h/t5_late_fusion_model_20260218_151904/test_predictions_late_fusion_inputGTmodal.json",
    "Flan-T5-Large-LoRA": "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_20260218_234219/test_predictions_late_fusion_inputGTmodal.json"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

rouge = evaluate.load("rouge")

# =========================
# Evaluation Function
# =========================

def evaluate_model(pred_file):

    with open(pred_file) as f:
        data = json.load(f)

    predictions = [d["prediction"] for d in data]
    references = [d["reference"] for d in data]

    # BLEU-4
    bleu = corpus_bleu(predictions, [references])
    bleu4 = bleu.score

    # ROUGE
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    # BERTScore
    P, R, F1 = bertscore(
        predictions,
        references,
        lang="en",
        device=DEVICE,
        verbose=False
    )

    bert_f1 = F1.mean().item()

    return {
        "BLEU-4": round(bleu4, 2),
        "ROUGE-1": round(rouge_result["rouge1"], 4),
        "ROUGE-L": round(rouge_result["rougeL"], 4),
        "BERTScore-F1": round(bert_f1, 4),
    }

# =========================
# Run Evaluation
# =========================

results = {}

for name, path in PREDICTION_FILES.items():
    print(f"\nEvaluating: {name}")
    metrics = evaluate_model(path)
    results[name] = metrics

# =========================
# Convert to DataFrame
# =========================

df = pd.DataFrame(results).T

# =========================
# Pretty Print Table
# =========================

print("\n================ Final Comparison ================\n")
print(df.to_string())
print("\n=================================================\n")

# =========================
# Save Results
# =========================
OUTPUT_DIR = "/home/woody/iwi5/iwi5325h/flan_t5_large_lora_20260218_234219"
df.to_csv(os.path.join(OUTPUT_DIR,"evaluation_results_modal_test_gt.csv"))

# Also save markdown table (论文可直接用)
with open(os.path.join(OUTPUT_DIR,"evaluation_results_modal_test_gt.md"), "w") as f:
    f.write(df.to_markdown())

print("Results saved to evaluation_results.csv and evaluation_results.md")
