# -*- coding: utf-8 -*-
"""
Created on 2026/5/18 14:40

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# -*- coding: utf-8 -*-

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
}

df = pd.DataFrame(data)

language_metrics = [
    "BLEU-2", "BLEU-4", "ROUGE-1", "METEOR", "BERTScore"
]

radgraph_metrics = [
    "RadGraph", "RadGraph-Entity", "RadGraph-Relation"
]

all_metrics = language_metrics + radgraph_metrics

# =========================
# Nice color palette
# =========================
colors = {
    "AutoRG-Brain Single-Modal": "#4C78A8",
    "AutoRG-Brain + T5-Small": "#F58518",
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "#54A24B",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "#E45756",
    "Qwen2.5-VL Zero-Shot": "#B279A2",
    "Qwen2.5-VL Few-Shot": "#9D755D",
}

# =========================
# Function: grouped bar chart
# =========================
def plot_grouped_bar(metrics, title, output_name):
    x = np.arange(len(metrics))
    width = 0.12

    plt.figure(figsize=(12, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        model = row["Model"]
        values = [row[m] for m in metrics]

        plt.bar(
            x + (i - len(df) / 2) * width + width / 2,
            values,
            width,
            label=model,
            color=colors[model],
            alpha=0.9
        )

    plt.xticks(x, metrics, fontsize=11)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=15, pad=15)

    plt.grid(axis="y", linestyle="--", alpha=0.35)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=9,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Figure 1: language metrics
# =========================
plot_grouped_bar(
    language_metrics,
    "Language Similarity Metrics Across Models",
    "language_metrics_bar.png"
)

# =========================
# Figure 2: RadGraph metrics
# =========================
plot_grouped_bar(
    radgraph_metrics,
    "Clinical Entity and Relation Metrics Across Models",
    "radgraph_metrics_bar.png"
)

# =========================
# Figure 3: normalized heatmap
# =========================
df_norm = df.copy()
for m in all_metrics:
    df_norm[m] = df[m] / df[m].max()

heatmap_data = df_norm.set_index("Model")[all_metrics]

plt.figure(figsize=(12, 5.5))
im = plt.imshow(heatmap_data.values, aspect="auto", cmap="YlGnBu")

plt.xticks(
    ticks=np.arange(len(all_metrics)),
    labels=all_metrics,
    rotation=35,
    ha="right",
    fontsize=10
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=heatmap_data.index,
    fontsize=10
)

plt.title("Normalized Metric Scores Across Models", fontsize=15, pad=15)

cbar = plt.colorbar(im)
cbar.set_label("Normalized Score", fontsize=11)

# Add values inside cells
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        value = heatmap_data.iloc[i, j]
        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="black"
        )

plt.tight_layout()
plt.savefig("normalized_metrics_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: language_metrics_bar.png")
print("Saved: radgraph_metrics_bar.png")
print("Saved: normalized_metrics_heatmap.png")