# -*- coding: utf-8 -*-
"""
Visualize how much late-fusion performance drops when modality inputs come
from the real AutoRG-Brain model instead of ground-truth modality reports.
"""

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "autorg_input_pipeline_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GT_INPUT_CSV = REPO_ROOT / "Autorg_output" / "evaluation_results.csv"
AUTORG_INPUT_CSV = REPO_ROOT / "text_to_text_result" / "evaluation_results_autorginput.csv"

FUSION_MODELS = [
    "AutoRG-Brain + T5-Small",
    "AutoRG-Brain + Flan-T5-Large (LoRA)",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)",
]

ALL_MODELS_AUTORG_INPUT = [
    "AutoRG-Brain Single-Modal",
    "AutoRG-Brain + T5-Small",
    "AutoRG-Brain + Flan-T5-Large (LoRA)",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)",
    "Qwen2.5-VL Zero-Shot",
    "Qwen2.5-VL Few-Shot",
]

METRIC_COLUMNS = [
    "BLEU-2",
    "BLEU-4",
    "ROUGE-1",
    "METEOR",
    "BERTScore",
    "RadGraph",
    "RadGraph-Entity",
    "RadGraph-Relation",
]

KEY_METRICS = [
    "BLEU-4",
    "ROUGE-1",
    "METEOR",
    "BERTScore",
    "RadGraph",
    "RadGraph-Entity",
]

COLORS = {
    "AutoRG-Brain Single-Modal": "#56B4E9",
    "AutoRG-Brain + T5-Small": "#0072B2",
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "#E69F00",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "#CC79A7",
    "Qwen2.5-VL Zero-Shot": "#009E73",
    "Qwen2.5-VL Few-Shot": "#117733",
}

SLOPE_COLORS = {
    "AutoRG-Brain + T5-Small": COLORS["AutoRG-Brain + T5-Small"],
    "AutoRG-Brain + Flan-T5-Large (LoRA)": COLORS["AutoRG-Brain + Flan-T5-Large (LoRA)"],
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": COLORS[
        "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)"
    ],
}

SLOPE_MARKERS = {
    "AutoRG-Brain + T5-Small": "o",
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "s",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "^",
}

SHORT_LABELS = {
    "AutoRG-Brain Single-Modal": "AutoRG-Brain\nSingle-Modal",
    "AutoRG-Brain + T5-Small": "AutoRG-Brain\n+ T5-Small",
    "AutoRG-Brain + Flan-T5-Large (LoRA)": "AutoRG-Brain\n+ Flan-T5-Large\n(LoRA)",
    "AutoRG-Brain + Flan-T5-Large (LoRA + PullLoss)": "AutoRG-Brain\n+ Flan-T5-Large\n(LoRA + PullLoss)",
    "Qwen2.5-VL Zero-Shot": "Qwen2.5-VL\nZero-Shot",
    "Qwen2.5-VL Few-Shot": "Qwen2.5-VL\nFew-Shot",
}


def load_metrics(path):
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Model"})
    df.columns = [c.replace(" ↑", "") for c in df.columns]
    for col in METRIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("Model")


def build_drop_table(gt_df, autorg_df):
    rows = []
    for model in FUSION_MODELS:
        for metric in METRIC_COLUMNS:
            gt_value = gt_df.loc[model, metric]
            autorg_value = autorg_df.loc[model, metric]
            absolute_drop = gt_value - autorg_value
            relative_drop = absolute_drop / gt_value * 100 if gt_value else np.nan
            retention = autorg_value / gt_value * 100 if gt_value else np.nan
            rows.append(
                {
                    "Model": model,
                    "Metric": metric,
                    "GT-input": gt_value,
                    "AutoRG-input": autorg_value,
                    "Absolute drop": absolute_drop,
                    "Relative drop (%)": relative_drop,
                    "Retention (%)": retention,
                }
            )
    return pd.DataFrame(rows)


def metric_to_percentage_points(metric, values):
    if metric.startswith("BLEU"):
        return values
    return values * 100


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)


def save_absolute_drop_chart(drop_df):
    fig, ax = plt.subplots(figsize=(14, 6.8))
    x = np.arange(len(METRIC_COLUMNS))
    width = 0.24

    for i, model in enumerate(FUSION_MODELS):
        subset = drop_df[drop_df["Model"] == model].set_index("Metric")
        drops = [
            metric_to_percentage_points(metric, subset.loc[metric, "Absolute drop"])
            for metric in METRIC_COLUMNS
        ]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            drops,
            width=width,
            color=COLORS[model],
            edgecolor="#333333",
            linewidth=0.8,
            label=model,
        )
        for bar, value in zip(bars, drops):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.6, max(drops) * 0.015),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#222222",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.16),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_COLUMNS, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Absolute drop (percentage points)", fontsize=12)
    ax.set_title(
        "Performance Drop Under Real AutoRG-Brain Input",
        fontsize=17,
        pad=14,
        fontweight="bold",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "autorg_input_absolute_drop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_retention_heatmap(drop_df):
    heat = (
        drop_df.pivot(index="Model", columns="Metric", values="Retention (%)")
        .loc[FUSION_MODELS, METRIC_COLUMNS]
    )

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    im = ax.imshow(heat.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(METRIC_COLUMNS)))
    ax.set_xticklabels(METRIC_COLUMNS, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(FUSION_MODELS)))
    ax.set_yticklabels([SHORT_LABELS[m] for m in FUSION_MODELS], fontsize=10)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            color = "white" if value < 45 else "#111111"
            ax.text(
                j,
                i,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )

    ax.set_title(
        "Metric Retention: AutoRG-Brain Input vs GT Modality Input",
        fontsize=17,
        pad=14,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Retention (%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "autorg_input_retention_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_slope_chart(gt_df, autorg_df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()

    for ax, metric in zip(axes, KEY_METRICS):
        label_rows = []
        for model in FUSION_MODELS:
            gt_value = metric_to_percentage_points(metric, gt_df.loc[model, metric])
            autorg_value = metric_to_percentage_points(metric, autorg_df.loc[model, metric])
            relative_drop = (gt_value - autorg_value) / gt_value * 100 if gt_value else np.nan

            ax.plot(
                [0, 1],
                [gt_value, autorg_value],
                marker=SLOPE_MARKERS[model],
                markersize=7.5,
                linewidth=3.0,
                color=SLOPE_COLORS[model],
                label=model,
            )
            label_text = (
                f"-{relative_drop:.0f}%"
                if relative_drop >= 0
                else f"+{abs(relative_drop):.0f}%"
            )
            label_rows.append(
                {
                    "model": model,
                    "value": autorg_value,
                    "text": label_text,
                    "color": SLOPE_COLORS[model],
                }
            )

        all_values = []
        for model in FUSION_MODELS:
            all_values.append(metric_to_percentage_points(metric, gt_df.loc[model, metric]))
            all_values.append(metric_to_percentage_points(metric, autorg_df.loc[model, metric]))
        value_min, value_max = min(all_values), max(all_values)
        value_range = max(value_max - value_min, 1.0)
        min_gap = value_range * 0.08

        label_rows.sort(key=lambda row: row["value"])
        placed = []
        for row in label_rows:
            y = row["value"]
            if placed and y - placed[-1] < min_gap:
                y = placed[-1] + min_gap
            placed.append(y)
        overflow = placed[-1] - (value_max + value_range * 0.10)
        if overflow > 0:
            placed = [y - overflow for y in placed]

        for row, label_y in zip(label_rows, placed):
            ax.annotate(
                row["text"],
                xy=(1, row["value"]),
                xytext=(1.08, label_y),
                textcoords="data",
                va="center",
                fontsize=11,
                color=row["color"],
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="-",
                    color=row["color"],
                    linewidth=1.0,
                    alpha=0.65,
                    shrinkA=0,
                    shrinkB=5,
                ),
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.82,
                    pad=0.2,
                ),
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["GT input", "AutoRG-Brain input"], fontsize=10)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.08, 1.30)
        ax.set_ylim(value_min - value_range * 0.12, value_max + value_range * 0.16)
        ax.set_ylabel("Score" if metric.startswith("BLEU") else "Score x 100")
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "From Oracle Modality Input to Real AutoRG-Brain Pipeline",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "gt_to_autorg_input_slope_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_realistic_pipeline_bar(autorg_df):
    metrics = ["BLEU-4", "ROUGE-1", "METEOR", "BERTScore", "RadGraph"]
    plot_df = autorg_df.loc[ALL_MODELS_AUTORG_INPUT, metrics].copy()
    for metric in metrics:
        plot_df[metric] = metric_to_percentage_points(metric, plot_df[metric])

    fig, ax = plt.subplots(figsize=(14, 6.6))
    x = np.arange(len(metrics))
    width = 0.12

    for i, model in enumerate(ALL_MODELS_AUTORG_INPUT):
        values = plot_df.loc[model, metrics].values
        offset = (i - (len(ALL_MODELS_AUTORG_INPUT) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[model],
            edgecolor="#333333",
            linewidth=0.75,
            label=model,
        )
        for j, (bar, value) in enumerate(zip(bars, values)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2 + (j % 2) * 0.9,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#111111",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.12),
                zorder=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score (BLEU as reported; other metrics x 100)", fontsize=12)
    ax.set_title(
        "Realistic Pipeline Performance with AutoRG-Brain Input",
        fontsize=17,
        pad=14,
        fontweight="bold",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "realistic_pipeline_autorg_input_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(drop_df):
    key = drop_df[drop_df["Metric"].isin(["BLEU-4", "ROUGE-1", "METEOR", "BERTScore"])]
    summary = (
        key.groupby("Model")[["Absolute drop", "Relative drop (%)", "Retention (%)"]]
        .mean()
        .sort_values("Relative drop (%)", ascending=False)
    )

    print("\nAverage drop over BLEU-4, ROUGE-1, METEOR, and BERTScore")
    print(summary.round(2).to_string())

    rouge = drop_df[drop_df["Metric"] == "ROUGE-1"].copy()
    rouge["Absolute drop"] = rouge["Absolute drop"] * 100
    print("\nROUGE-1 drop in percentage points")
    print(rouge[["Model", "Absolute drop", "Relative drop (%)", "Retention (%)"]].round(2).to_string(index=False))


def main():
    gt_df = load_metrics(GT_INPUT_CSV)
    autorg_df = load_metrics(AUTORG_INPUT_CSV)
    drop_df = build_drop_table(gt_df, autorg_df)

    drop_df.to_csv(OUTPUT_DIR / "autorg_input_drop_summary.csv", index=False)

    save_absolute_drop_chart(drop_df)
    save_retention_heatmap(drop_df)
    save_slope_chart(gt_df, autorg_df)
    save_realistic_pipeline_bar(autorg_df)
    print_summary(drop_df)

    print(f"\nSaved figures and summary to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
