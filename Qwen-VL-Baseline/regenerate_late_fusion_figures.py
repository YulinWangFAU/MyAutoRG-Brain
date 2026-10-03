# -*- coding: utf-8 -*-
"""
Regenerate the main late-fusion figures with one consistent color palette,
larger bold labels, and collision-aware annotations.
"""

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


GT_INPUT_CSV = REPO_ROOT / "Autorg_output" / "evaluation_results.csv"
AUTORG_INPUT_CSV = REPO_ROOT / "text_to_text_result" / "evaluation_results_autorginput.csv"

MODEL_ORDER = [
    "AutoRG-Brain Single-Modal",
    "AutoRG-Brain + T5-Small",
    "AutoRG-Brain + Flan-T5-Large (LoRA)",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)",
    "Qwen2.5-VL Zero-Shot",
    "Qwen2.5-VL Few-Shot",
]

MODEL_COLORS = {
    "AutoRG-Brain Single-Modal": "#56B4E9",
    "AutoRG-Brain + T5-Small": "#0072B2",
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "#E69F00",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "#CC79A7",
    "Qwen2.5-VL Zero-Shot": "#009E73",
    "Qwen2.5-VL Few-Shot": "#117733",
}

MODEL_MARKERS = {
    "AutoRG-Brain Single-Modal": "o",
    "AutoRG-Brain + T5-Small": "o",
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "s",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "^",
    "Qwen2.5-VL Zero-Shot": "D",
    "Qwen2.5-VL Few-Shot": "P",
}

LANGUAGE_METRICS = ["BLEU-2", "BLEU-4", "ROUGE-1", "METEOR", "BERTScore"]
RADGRAPH_SOURCE_METRICS = ["RadGraph", "RadGraph-Entity", "RadGraph-Relation"]
RADGRAPH_DISPLAY_LABELS = {
    "RadGraph": "RG-E",
    "RadGraph-Entity": "RG-ER",
    "RadGraph-Relation": "RG-ĒR",
}
ALL_METRICS = LANGUAGE_METRICS + RADGRAPH_SOURCE_METRICS


def load_metrics(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Model"})
    df.columns = [c.replace(" ↑", "") for c in df.columns]
    for metric in ALL_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df.set_index("Model").loc[MODEL_ORDER].reset_index()


def display_value(metric, value):
    return value if metric.startswith("BLEU") else value * 100


def axis_label(metrics):
    if all(m.startswith("BLEU") for m in metrics):
        return "BLEU score"
    if not any(m.startswith("BLEU") for m in metrics):
        return "Score x 100"
    return "Score (BLEU as reported; other metrics x 100)"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.32)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=12)


def add_grouped_bar_labels(ax, bars, values, y_max, fmt="{:.1f}", fontsize=12):
    min_gap = y_max * 0.035
    for idx, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        x_shift = ((idx % 3) - 1) * bar.get_width() * 0.18
        y_shift = y_max * (0.018 + 0.010 * (idx % 2))
        y = min(height + y_shift, y_max - min_gap)
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_shift,
            y,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=0.18),
            zorder=20,
        )


def save_grouped_bar(df, metrics, title, output_names, figsize=(18, 7.5), legend_cols=3):
    plot_df = df.copy()
    x = np.arange(len(metrics))
    width = 0.12

    values_by_model = {
        row["Model"]: [display_value(metric, row[metric]) for metric in metrics]
        for _, row in plot_df.iterrows()
    }
    y_max = max(max(values) for values in values_by_model.values()) * 1.22

    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(MODEL_ORDER):
        values = values_by_model[model]
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="#222222",
            linewidth=0.8,
            label=model,
            zorder=3,
        )
        add_grouped_bar_labels(ax, bars, values, y_max, fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [RADGRAPH_DISPLAY_LABELS.get(metric, metric) for metric in metrics],
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel(axis_label(metrics), fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=16)
    ax.set_ylim(0, y_max)
    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=legend_cols,
        frameon=False,
        fontsize=11,
    )
    fig.tight_layout()
    for output_name in output_names:
        fig.savefig(SCRIPT_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(df, title, output_names):
    norm_df = df.copy()
    for metric in ALL_METRICS:
        norm_df[metric] = norm_df[metric] / norm_df[metric].max()
    heat = norm_df.set_index("Model")[ALL_METRICS]

    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(ALL_METRICS)))
    ax.set_xticklabels(
        [RADGRAPH_DISPLAY_LABELS.get(metric, metric) for metric in ALL_METRICS],
        rotation=30,
        ha="right",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=16)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if value >= 0.62 else "#111111",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Normalized score", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    for output_name in output_names:
        fig.savefig(SCRIPT_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_radar(df, title, output_names):
    norm_df = df.copy()
    for metric in ALL_METRICS:
        norm_df[metric] = norm_df[metric] / norm_df[metric].max()

    angles = np.linspace(0, 2 * np.pi, len(ALL_METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(11, 8.8))
    ax = plt.subplot(111, polar=True)

    for _, row in norm_df.iterrows():
        model = row["Model"]
        values = [row[metric] for metric in ALL_METRICS]
        values += values[:1]
        ax.plot(
            angles,
            values,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=4.5,
            linewidth=2.5,
            label=model,
        )
        ax.fill(angles, values, color=MODEL_COLORS[model], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [RADGRAPH_DISPLAY_LABELS.get(metric, metric) for metric in ALL_METRICS],
        fontsize=11,
        fontweight="bold",
    )
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=19, fontweight="bold", pad=22)
    ax.grid(alpha=0.34)
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.10), fontsize=10, frameon=False)
    fig.tight_layout()
    for output_name in output_names:
        fig.savefig(SCRIPT_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def regenerate_set(df, prefix):
    if prefix == "gt":
        suffix_outputs = {
            "language": ["language_metrics_dual_axis_bar_pipeline_names.png"],
            "radgraph": ["radgraph_metrics_bar_pipeline_names.png"],
            "heatmap": ["normalized_metrics_heatmap_pipeline_names.png"],
            "radar": ["normalized_metrics_radar_pipeline_names.png"],
        }
        title_suffix = "GT Modality Input"
    else:
        suffix_outputs = {
            "language": ["1language_metrics_dual_axis_bar_pipeline_name1s.png"],
            "radgraph": ["1radgraph_metrics_bar_pipeline_name1s.png"],
            "heatmap": ["1normalized_metrics_heatmap_pipeline_names.png"],
            "radar": ["1normalized_metrics_radar_pipeline_name1s.png"],
        }
        title_suffix = "Real AutoRG-Brain Input"

    save_grouped_bar(
        df,
        LANGUAGE_METRICS,
        f"Language Metrics Across Models ({title_suffix})",
        suffix_outputs["language"],
        figsize=(18.5, 8.0),
    )
    save_grouped_bar(
        df,
        RADGRAPH_SOURCE_METRICS,
        f"F1-RadGraph Reward Levels ({title_suffix})",
        suffix_outputs["radgraph"],
        figsize=(15.5, 7.8),
    )
    save_heatmap(
        df,
        f"Normalized Metric Scores Across Models ({title_suffix})",
        suffix_outputs["heatmap"],
    )
    save_radar(
        df,
        f"Normalized Evaluation Metrics ({title_suffix})",
        suffix_outputs["radar"],
    )


def main():
    gt_df = load_metrics(GT_INPUT_CSV)
    autorg_df = load_metrics(AUTORG_INPUT_CSV)

    regenerate_set(gt_df, "gt")
    regenerate_set(autorg_df, "autorg")

    print("Regenerated GT-input and AutoRG-input main figures with consistent colors.")


if __name__ == "__main__":
    main()
