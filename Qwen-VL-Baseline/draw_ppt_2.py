# -*- coding: utf-8 -*-
"""
Created on 2026/5/19 16:34

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
Created on 2026/5/18 14:40

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
    "BERTScore": [0.3951, 0.6760, 0.5735, 0.5616, 0.0914, 0.3771],
    "RadGraph": [0.3752, 0.3778, 0.2646, 0.2646, 0.1090, 0.1458],
    "RadGraph-Entity": [0.2885, 0.5427, 0.5657, 0.5747, 0.1532, 0.2500],
    "RadGraph-Relation": [0.3467, 0.2593, 0.4911, 0.5337, 0.0593, 0.1926],
}

df = pd.DataFrame(data)

# =========================
# Metric groups
# =========================
bleu_metrics = ["BLEU-2", "BLEU-4"]
score01_metrics = ["ROUGE-1", "METEOR", "BERTScore"]
language_metrics = bleu_metrics + score01_metrics

radgraph_metrics = [
    "RadGraph",
    "RadGraph-Entity",
    "RadGraph-Relation",
]

all_metrics = language_metrics + radgraph_metrics


# =========================
# Pastel grouped color palette
# Similar model families use similar colors
# =========================
colors = {
    # Grounded baseline
    "AutoRG-Brain Single-Modal": "#8ECAE6",

    # T5 / Flan-T5 family
    "AutoRG-Brain + T5-Small": "#FFB4A2",
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "#F6BD60",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "#E5989B",

    # Qwen VLM family
    "Qwen2.5-VL Zero-Shot": "#CDB4DB",
    "Qwen2.5-VL Few-Shot": "#FFC8DD",
}


# =========================
# Figure 1: Normalized radar chart
# =========================
df_norm = df.copy()
for m in all_metrics:
    df_norm[m] = df[m] / df[m].max()

categories = all_metrics
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

for _, row in df_norm.iterrows():
    model = row["Model"]
    values = [row[m] for m in categories]
    values += values[:1]

    ax.plot(
        angles,
        values,
        label=model,
        color=colors[model],
        linewidth=3
    )

    ax.fill(
        angles,
        values,
        color=colors[model],
        alpha=0.18
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)

ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=10)

ax.set_title(
    "Normalized Evaluation Metrics Across Models",
    fontsize=18,
    pad=22
)

ax.grid(alpha=0.35)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.05, 1.10),
    fontsize=10,
    frameon=True
)

plt.tight_layout()
plt.savefig(
    "normalized_metrics_radar_pastel.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# Figure 2: Language metrics with dual y-axis
# BLEU uses left y-axis
# ROUGE / METEOR / BERTScore use right y-axis
# =========================
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

models = df["Model"].tolist()

x_bleu = np.arange(len(bleu_metrics))
x_score01 = np.arange(len(score01_metrics)) + len(bleu_metrics) + 1.0

width = 0.11

for i, (_, row) in enumerate(df.iterrows()):
    model = row["Model"]
    offset = (i - len(df) / 2) * width + width / 2

    ax1.bar(
        x_bleu + offset,
        [row[m] for m in bleu_metrics],
        width,
        color=colors[model],
        edgecolor="#444444",
        linewidth=1.0,
        alpha=0.95
    )

    ax2.bar(
        x_score01 + offset,
        [row[m] for m in score01_metrics],
        width,
        color=colors[model],
        edgecolor="#444444",
        linewidth=1.0,
        alpha=0.95
    )

xticks = list(x_bleu) + list(x_score01)
xlabels = bleu_metrics + score01_metrics

ax1.set_xticks(xticks)
ax1.set_xticklabels(xlabels, fontsize=12)

ax1.set_ylabel("BLEU Score", fontsize=13)
ax2.set_ylabel("ROUGE / METEOR / BERTScore", fontsize=13)

ax1.set_ylim(0, 65)
ax2.set_ylim(0, 0.85)

ax1.tick_params(axis="y", labelsize=11)
ax2.tick_params(axis="y", labelsize=11)

ax1.grid(axis="y", linestyle="--", alpha=0.3)

ax1.set_title(
    "Language Similarity Metrics Across Models",
    fontsize=17,
    pad=15
)

legend_handles = [
    Patch(
        facecolor=colors[m],
        edgecolor="#444444",
        label=m
    )
    for m in models
]

ax1.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=10,
    frameon=False
)

plt.tight_layout()
plt.savefig(
    "language_metrics_dual_axis_bar.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# Function: grouped bar chart
# =========================
def plot_grouped_bar(metrics, title, output_name, ylabel="Score", ylim=None):
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
            edgecolor="#444444",
            linewidth=1.0,
            alpha=0.95
        )

    plt.xticks(x, metrics, fontsize=12)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=17, pad=15)

    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=10,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Figure 3: RadGraph metrics
# =========================
plot_grouped_bar(
    radgraph_metrics,
    "Clinical Entity and Relation Metrics Across Models",
    "radgraph_metrics_bar_pastel.png",
    ylabel="Score",
    ylim=(0, 0.65)
)


# =========================
# Figure 4: Normalized heatmap
# Larger font + adaptive text color
# =========================
heatmap_data = df_norm.set_index("Model")[all_metrics]

plt.figure(figsize=(13, 6.2))

im = plt.imshow(
    heatmap_data.values,
    aspect="auto",
    cmap="YlGnBu",
    vmin=0,
    vmax=1
)

plt.xticks(
    ticks=np.arange(len(all_metrics)),
    labels=all_metrics,
    rotation=35,
    ha="right",
    fontsize=12
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=heatmap_data.index,
    fontsize=12
)

plt.title(
    "Normalized Metric Scores Across Models",
    fontsize=18,
    pad=18
)

cbar = plt.colorbar(im)
cbar.set_label("Normalized Score", fontsize=13)
cbar.ax.tick_params(labelsize=11)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        value = heatmap_data.iloc[i, j]

        text_color = "white" if value >= 0.65 else "black"

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=text_color
        )

plt.tight_layout()
plt.savefig(
    "normalized_metrics_heatmap_readable.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


print("Saved: normalized_metrics_radar_pastel2.png")
print("Saved: language_metrics_dual_axis_bar2.png")
print("Saved: radgraph_metrics_bar_pastel2.png")
print("Saved: normalized_metrics_heatmap_readable2.png")