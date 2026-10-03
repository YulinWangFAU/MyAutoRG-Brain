# -*- coding: utf-8 -*-
"""
Created on 2026/5/10 18:20

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Data
# =========================
data = {
    "Model": [
        "AutoRG-Brain Single-Modal",
        "AutoRG-Brain + T5-Small",
        "AutoRG-Brain + Flan-T5-Large (LoRA)",
        "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)",
        "Qwen2.5-VL Zero-Shot",
        "Qwen2.5-VL Few-Shot",
    ],
    "BLEU-2": [27.03, 58.26, 48.91, 46.86, 9.73, 27.93],
    "BLEU-4": [13.36, 44.83, 35.18, 33.57, 1.99, 13.82],
    "ROUGE-1": [0.4414, 0.7531, 0.6763, 0.6686, 0.2768, 0.4845],
    "METEOR": [0.3656, 0.6787, 0.5717, 0.5552, 0.2043, 0.3314],
    "BERTScore": [0.3951, 0.676, 0.5735, 0.5616, 0.0914, 0.3771],
    "RadGraph": [0.3752, 0.3778, 0.2646, 0.2646, 0.109, 0.1458],
    "RadGraph-Entity": [0.2885, 0.5427, 0.5657, 0.5747, 0.1532, 0.25],
    "RadGraph-Relation": [0.3467, 0.2593, 0.4911, 0.5337, 0.0593, 0.1926],
    "RadGraph 95% CI": [
        "[0.3636, 0.3810]",
        "[0.3333, 0.4667]",
        "[0.1765, 0.3750]",
        "[0.1765, 0.3750]",
        "[0.0435, 0.1905]",
        "[0.0625, 0.2500]",
    ],
}

df = pd.DataFrame(data)

metrics = [
    "BLEU-2",
    "BLEU-4",
    "ROUGE-1",
    "METEOR",
    "BERTScore",
    "RadGraph",
    "RadGraph-Entity",
    "RadGraph-Relation",
]

# =========================
# Normalize each metric to 0-1
# =========================
df_norm = df.copy()

for m in metrics:
    df_norm[m] = df[m] / df[m].max()

# =========================
# Radar chart
# =========================
N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(10, 7))
ax = plt.subplot(111, polar=True)

for _, row in df_norm.iterrows():
    values = row[metrics].tolist()
    values += values[:1]

    ax.plot(angles, values, linewidth=2, label=row["Model"])
    ax.fill(angles, values, alpha=0.10)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)

ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=9)

ax.set_title("Normalized Evaluation Metrics Across Models", fontsize=15, pad=25)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.45, 1.15),
    fontsize=9
)

plt.tight_layout()
plt.savefig("model_radar_chart_with_autorg.png", dpi=300, bbox_inches="tight")

# Optional: save table
df.to_csv("model_metrics_with_autorg.csv", index=False)

print("Saved: model_radar_chart_with_autorg.png")
print("Saved: model_metrics_with_autorg.csv")