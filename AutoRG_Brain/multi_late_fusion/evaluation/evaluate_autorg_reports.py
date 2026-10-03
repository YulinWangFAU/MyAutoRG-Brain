# -*- coding: utf-8 -*-
"""
Created on 2026/3/10 11:43

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import json
import pandas as pd
from collections import defaultdict
from sacrebleu.metrics import BLEU
import evaluate
from bert_score import score as bertscore

# =========================
# Paths
# =========================

DATASET_PATH = "evaluation_dataset.json"

PRED_FILES = {
    "train": "autorg_predictions_train.json",
    "val": "autorg_predictions_val.json",
    "test": "autorg_predictions_test.json"
}

# =========================
# Load dataset
# =========================

dataset = json.load(open(DATASET_PATH))

print("Dataset size:", len(dataset))

# =========================
# Load predictions
# =========================

preds = {}

for split, path in PRED_FILES.items():
    data = json.load(open(path))
    preds.update(data)

print("Prediction size:", len(preds))

# =========================
# Metrics
# =========================

bleu = BLEU()
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


def compute_metrics(gt_list, pred_list):

    bleu_score = bleu.corpus_score(pred_list, [gt_list])

    rouge_score = rouge.compute(
        predictions=pred_list,
        references=gt_list
    )

    meteor_score = meteor.compute(
        predictions=pred_list,
        references=gt_list
    )

    P, R, F1 = bertscore(
        pred_list,
        gt_list,
        lang="en",
        rescale_with_baseline=True
    )

    return {
        "BLEU2": bleu_score.precisions[1],
        "BLEU4": bleu_score.score,
        "ROUGE1": rouge_score["rouge1"],
        "METEOR": meteor_score["meteor"],
        "BERTScore": float(F1.mean())
    }


# =========================
# Grouping helper
# =========================

def evaluate_group(data):

    gt = []
    pred = []

    for item in data:

        key = item["key"]

        if key not in preds:
            continue

        gt.append(item["gt"])
        pred.append(preds[key])

    if len(gt) == 0:
        return None

    return compute_metrics(gt, pred)


# =========================
# 1 Overall
# =========================

overall = evaluate_group(dataset)

print("\n===== OVERALL =====")

for k, v in overall.items():
    print(k, round(v, 4))


# =========================
# 2 Split
# =========================

split_groups = defaultdict(list)

for item in dataset:
    split_groups[item["split"]].append(item)

split_results = {}

print("\n===== SPLIT =====")

for split, data in split_groups.items():

    res = evaluate_group(data)

    split_results[split] = res

    print("\n", split)

    for k, v in res.items():
        print(k, round(v, 4))


# =========================
# 3 Modality
# =========================

modal_groups = defaultdict(list)

for item in dataset:
    modal_groups[item["modal"]].append(item)

modal_results = {}

print("\n===== MODALITY =====")

for modal, data in modal_groups.items():

    res = evaluate_group(data)

    modal_results[modal] = res

    print("\n", modal)

    for k, v in res.items():
        print(k, round(v, 4))


# =========================
# 4 Dataset (GLI vs MEN)
# =========================

dataset_groups = defaultdict(list)

for item in dataset:
    dataset_groups[item["dataset"]].append(item)

dataset_results = {}

print("\n===== DATASET =====")

for name, data in dataset_groups.items():

    res = evaluate_group(data)

    dataset_results[name] = res

    print("\n", name)

    for k, v in res.items():
        print(k, round(v, 4))


# =========================
# Save CSV
# =========================

def dict_to_df(results):

    rows = []

    for name, metrics in results.items():

        row = {"group": name}

        for k, v in metrics.items():
            row[k] = v

        rows.append(row)

    return pd.DataFrame(rows)


dict_to_df(split_results).to_csv("metrics_by_split.csv", index=False)
dict_to_df(modal_results).to_csv("metrics_by_modality.csv", index=False)
dict_to_df(dataset_results).to_csv("metrics_by_dataset.csv", index=False)

print("\nSaved CSV files.")