# -*- coding: utf-8 -*-

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

# =========================
# Paths
# =========================

BASE_PATH = "/home/woody/iwi5/iwi5325h/autorg_predictions/"

DATASET_PATH = BASE_PATH + "evaluation_dataset.json"

PRED_FILES = {
    "train": BASE_PATH + "autorg_predictions_train.json",
    "val": BASE_PATH + "autorg_predictions_val.json",
    "test": BASE_PATH + "autorg_predictions_test.json"
}

# =========================
# Config
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOOTSTRAP_SAMPLES = 1000
BERT_BATCH_SIZE = 16

print("Device:", DEVICE)

# =========================
# Load metrics
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
# Utils
# =========================

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bootstrap_ci(values):

    values = np.array(values)
    means = []

    for _ in range(BOOTSTRAP_SAMPLES):

        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    return np.percentile(means, 2.5), np.percentile(means, 97.5)


# =========================
# Load data
# =========================

dataset = json.load(open(DATASET_PATH))

preds = {}

for split, path in PRED_FILES.items():
    preds.update(json.load(open(path)))

print("Dataset:", len(dataset))
print("Predictions:", len(preds))

# =========================
# Metric function
# =========================

def compute_metrics(predictions, references):

    bleu2 = bleu2_metric.corpus_score(predictions, [references]).score
    bleu4 = bleu4_metric.corpus_score(predictions, [references]).score

    rouge1 = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )["rouge1"]

    meteor_score = meteor.compute(
        predictions=predictions,
        references=references
    )["meteor"]

    P, R, F1 = bertscore(
        predictions,
        references,
        lang="en",
        device=DEVICE,
        batch_size=BERT_BATCH_SIZE,
        rescale_with_baseline=True
    )

    bert_f1 = F1.mean().item()

    clean_memory()

    mean_scores, reward_list, *_ = f1radgraph(
        hyps=predictions,
        refs=references
    )

    reward_array = np.array(reward_list)

    per_sample_overall = reward_array[:, 0]
    per_sample_entity = reward_array[:, 1]
    per_sample_relation = reward_array[:, 2]

    overall_f1 = np.mean(per_sample_overall)
    entity_f1 = np.mean(per_sample_entity)
    relation_f1 = np.mean(per_sample_relation)

    ci_low, ci_high = bootstrap_ci(per_sample_overall)

    return {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "ROUGE-1": rouge1,
        "METEOR": meteor_score,
        "BERTScore": bert_f1,
        "RadGraph": overall_f1,
        "RadGraph-Entity": entity_f1,
        "RadGraph-Relation": relation_f1,
        "RadGraph-CI": f"[{ci_low:.4f},{ci_high:.4f}]"
    }


# =========================
# Build groups
# =========================

pairs = []
split_groups = {}
modal_groups = {}
dataset_groups = {}

for item in dataset:

    key = item["key"]

    if key not in preds:
        continue

    pred = preds[key]
    ref = item["gt"]

    pairs.append((pred, ref))

    split = item["split"]
    modal = item["modal"]
    ds = item["dataset"]

    split_groups.setdefault(split, []).append((pred, ref))
    modal_groups.setdefault(modal, []).append((pred, ref))
    dataset_groups.setdefault(ds, []).append((pred, ref))


# =========================
# 1 MAIN RESULTS
# =========================

print("\nComputing MAIN results")

preds_all = [p for p, r in pairs]
refs_all = [r for p, r in pairs]

main_metrics = compute_metrics(preds_all, refs_all)

main_table = pd.DataFrame([main_metrics], index=["AutoRG"])

# =========================
# 2 SPLIT RESULTS
# =========================

print("\nComputing SPLIT results")

split_results = {}

for split, data in split_groups.items():

    preds_s = [p for p, r in data]
    refs_s = [r for p, r in data]

    split_results[split] = compute_metrics(preds_s, refs_s)

split_table = pd.DataFrame(split_results).T

# =========================
# 3 MODALITY RESULTS
# =========================

print("\nComputing MODALITY results")

modal_results = {}

for modal, data in modal_groups.items():

    preds_m = [p for p, r in data]
    refs_m = [r for p, r in data]

    modal_results[modal] = compute_metrics(preds_m, refs_m)

modal_table = pd.DataFrame(modal_results).T

# =========================
# 4 DATASET RESULTS
# =========================

print("\nComputing DATASET results")

dataset_results = {}

for ds, data in dataset_groups.items():

    preds_d = [p for p, r in data]
    refs_d = [r for p, r in data]

    dataset_results[ds] = compute_metrics(preds_d, refs_d)

dataset_table = pd.DataFrame(dataset_results).T

# =========================
# Save tables
# =========================

os.makedirs(BASE_PATH + "evaluation_results", exist_ok=True)

main_table.to_csv(BASE_PATH + "evaluation_results/main_results.csv")
split_table.to_csv(BASE_PATH + "evaluation_results/split_results.csv")
modal_table.to_csv(BASE_PATH + "evaluation_results/modality_analysis.csv")
dataset_table.to_csv(BASE_PATH + "evaluation_results/dataset_analysis.csv")

print("\nSaved results to evaluation_results/")
# =========================
# Generate LaTeX tables
# =========================

def dataframe_to_latex(df, caption, label):

    latex = df.round(4).to_latex(
        index=True,
        escape=False
    )

    latex = f"""
\\begin{{table}}[ht]
\\centering
{latex}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""

    return latex


latex_main = dataframe_to_latex(
    main_table,
    "Automatic report generation performance.",
    "tab:main_results"
)

latex_modal = dataframe_to_latex(
    modal_table,
    "Performance across MRI modalities.",
    "tab:modal_results"
)

latex_dataset = dataframe_to_latex(
    dataset_table,
    "Performance across datasets.",
    "tab:dataset_results"
)

with open(BASE_PATH + "evaluation_results/main_results_latex.txt","w") as f:
    f.write(latex_main)

with open(BASE_PATH + "evaluation_results/modality_latex.txt","w") as f:
    f.write(latex_modal)

with open(BASE_PATH + "evaluation_results/dataset_latex.txt","w") as f:
    f.write(latex_dataset)

print("LaTeX tables generated.")