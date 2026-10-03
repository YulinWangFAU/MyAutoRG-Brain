# -*- coding: utf-8 -*-
"""
Created on 2026/5/19 16:49

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
    "BLEU-2": [27.03, 35.93, 31.00, 29.39, 9.73, 27.93],
    "BLEU-4": [13.36, 19.87, 16.15, 15.18, 1.99, 13.82],
    "ROUGE-1": [0.4414, 0.5473, 0.5014, 0.4997, 0.2768, 0.4845],
    "METEOR": [0.3656, 0.4311, 0.3659, 0.3569, 0.2043, 0.3314],
    "BERTScore": [0.3951, 0.4692, 0.3934, 0.3957, 0.0914, 0.3771],
    "RadGraph": [0.3752, 0.3740, 0.3010, 0.3010, 0.1090, 0.1458],
    "RadGraph-Entity": [0.2885, 0.2892, 0.3157, 0.3157, 0.1532, 0.2500],
    "RadGraph-Relation": [0.3467, 0.2016, 0.1111, 0.0926, 0.0593, 0.1926],
    "RadGraph 95% CI": [
        "[0.3636, 0.3810]",
        "[0.2927, 0.4878]",
        "[0.2439, 0.3590]",
        "[0.2439, 0.3590]",
        "[0.0435, 0.1905]",
        "[0.0625, 0.2500]",
    ],
}

df = pd.DataFrame(data)

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
# Stronger pastel color palette
# =========================
colors = {
    # "AutoRG-Brain Single-Modal": "#6BAED6",
    # "AutoRG-Brain + T5-Small": "#FF9F80",
    # "AutoRG-Brain + Flan-T5-Large (LoRA)": "#F4A62A",
    # "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "#D96C75",
    # "Qwen2.5-VL Zero-Shot": "#B79AD0",
    # "Qwen2.5-VL Few-Shot": "#F49AC2",

    "AutoRG-Brain Single-Modal": "#A8D8EA",                      # 柔和浅蓝
    "AutoRG-Brain + T5-Small": "#FFB7A5",                        # 马卡龙蜜桃橙
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "#FFD89C",            # 奶油杏黄
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "#F3A6B7", # 低饱和玫瑰粉
    "Qwen2.5-VL Zero-Shot": "#B8E0D2",  # 薄荷绿
    "Qwen2.5-VL Few-Shot": "#79C9B8",  # 青绿色
}



# =========================
# Normalize for radar / heatmap
# =========================
df_norm = df.copy()
for m in all_metrics:
    df_norm[m] = df[m] / df[m].max()


# =========================
# Helper: add value labels on bars
# =========================
def add_bar_labels(
    ax,
    bars,
    fmt="{:.2f}",
    fontsize=10,
    rotation=0,
    x_shift_scale=0.018,
    y_shift_scale=0.018,
    y_shift_min=0.008
):
    """
    Improved value labels:
    - bold
    - horizontal
    - alternating left/right positions
    - white background for readability
    - lifted above bar top
    """

    for idx, bar in enumerate(bars):
        height = bar.get_height()

        if height == 0:
            continue

        # Horizontal stagger: left, center, right pattern
        pattern = [-1, 0, 1]
        x_shift = pattern[idx % 3] * x_shift_scale

        # Lift label above bar
        y_shift = max(height * y_shift_scale, y_shift_min)

        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_shift,
            height + y_shift,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            rotation=rotation,
            color="#111111",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                pad=0.25
            ),
            zorder=10
        )


# =========================
# Figure 1: Radar chart
# =========================
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
        linewidth=3.2
    )

    ax.fill(
        angles,
        values,
        color=colors[model],
        alpha=0.10
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
    fontsize=9,
    frameon=True
)

plt.tight_layout()
plt.savefig(
    "1normalized_metrics_radar_pipeline_name1s.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# Figure 2: Language metrics with dual y-axis + value labels
# =========================
fig, ax1 = plt.subplots(figsize=(16, 7))
ax2 = ax1.twinx()

models = df["Model"].tolist()

x_bleu = np.arange(len(bleu_metrics))
x_score01 = np.arange(len(score01_metrics)) + len(bleu_metrics) + 1.0

width = 0.11

for i, (_, row) in enumerate(df.iterrows()):
    model = row["Model"]
    offset = (i - len(df) / 2) * width + width / 2

    bars1 = ax1.bar(
        x_bleu + offset,
        [row[m] for m in bleu_metrics],
        width,
        color=colors[model],
        edgecolor="#333333",
        linewidth=0.9,
        alpha=1.0,
        zorder=3
    )

    bars2 = ax2.bar(
        x_score01 + offset,
        [row[m] for m in score01_metrics],
        width,
        color=colors[model],
        edgecolor="#333333",
        linewidth=0.9,
        alpha=1.0,
        zorder=3
    )

    add_bar_labels(
        ax1,
        bars1,
        fmt="{:.1f}",
        fontsize=10,
        rotation=0,
        x_shift_scale=0.018,
        y_shift_scale=0.015,
        y_shift_min=0.8
    )

    add_bar_labels(
        ax2,
        bars2,
        fmt="{:.2f}",
        fontsize=10,
        rotation=0,
        x_shift_scale=0.018,
        y_shift_scale=0.020,
        y_shift_min=0.012
    )

xticks = list(x_bleu) + list(x_score01)
xlabels = bleu_metrics + score01_metrics

ax1.set_xticks(xticks)
ax1.set_xticklabels(xlabels, fontsize=13)

ax1.set_ylabel("BLEU Score", fontsize=14)
ax2.set_ylabel("ROUGE / METEOR / BERTScore", fontsize=14)

ax1.set_ylim(0, 72)
ax2.set_ylim(0, 0.95)

ax1.tick_params(axis="y", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)

ax1.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

ax1.set_title(
    "Language Similarity Metrics Across Models",
    fontsize=18,
    pad=16
)

legend_handles = [
    Patch(
        facecolor=colors[m],
        edgecolor="#333333",
        label=m
    )
    for m in models
]

ax1.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
    fontsize=10,
    frameon=False
)

plt.tight_layout()
plt.savefig(
    "1language_metrics_dual_axis_bar_pipeline_name1s.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# =========================
# Figure 3: RadGraph metrics + value labels
# =========================
def plot_grouped_bar(metrics, title, output_name, ylabel="Score", ylim=None):
    x = np.arange(len(metrics))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (_, row) in enumerate(df.iterrows()):
        model = row["Model"]
        values = [row[m] for m in metrics]
        offset = (i - len(df) / 2) * width + width / 2

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model,
            color=colors[model],
            edgecolor="#333333",
            linewidth=0.9,
            alpha=1.0,
            zorder=3
        )

        add_bar_labels(
            ax,
            bars,
            fmt="{:.2f}",
            fontsize=10,
            rotation=0,
            x_shift_scale=0.020,
            y_shift_scale=0.030,
            y_shift_min=0.012
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=18, pad=16)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=10,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()


plot_grouped_bar(
    radgraph_metrics,
    "Clinical Entity and Relation Metrics Across Models",
    "1radgraph_metrics_bar_pipeline_name1s.png",
    ylabel="Score",
    ylim=(0, 0.72)
)


# =========================
# Figure 4: Normalized heatmap
# =========================
heatmap_data = df_norm.set_index("Model")[all_metrics]

plt.figure(figsize=(14, 6.5))

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
    fontsize=11
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
    "1normalized_metrics_heatmap_pipeline_names.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


print("Saved: 1normalized_metrics_radar_pipeline_names.png")
print("Saved: 1language_metrics_dual_axis_bar_pipeline_names.png")
print("Saved: 1radgraph_metrics_bar_pipeline_names.png")
print("Saved: 1normalized_metrics_heatmap_pipeline_names.png")