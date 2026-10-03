# -*- coding: utf-8 -*-
"""
Created on 2026/5/5 00:47

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
splits = ["Training", "Validation", "Test"]
gli = np.array([644, 92, 184])
men = np.array([644, 92, 184])

totals = gli + men
subjects = totals // 4

# percentage
total_all = totals.sum()
split_percentages = totals / total_all * 100

# -----------------------------
# Colors（更干净）
# -----------------------------
gli_color = "#F4D06F"   # yellow
men_color = "#BFA2DB"   # purple

x = np.arange(len(splits))

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))

# bars
ax.bar(x, gli, label="BraTS-GLI", color=gli_color, edgecolor="white")
ax.bar(x, men, bottom=gli, label="BraTS-MEN", color=men_color, edgecolor="white")

# -----------------------------
# Top labels（只保留这一层！）
# -----------------------------
for i in range(len(splits)):
    ax.text(
        x[i],
        totals[i] + 40,
        f"{totals[i]} volumes\n{subjects[i]} subjects\n({split_percentages[i]:.1f}%)",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

# -----------------------------
# Style（论文风格）
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=12, fontweight="bold")

ax.set_ylabel("Number of Volumes", fontsize=11)
ax.set_title(
    "RadGenome-Brain MRI Split and Subdataset Composition",
    fontsize=14,
    fontweight="bold"
)

# grid
ax.yaxis.grid(True, linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# remove border
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend
ax.legend(frameon=False, fontsize=11)

# footnote（很重要）
plt.figtext(
    0.5, -0.02,
    "Subject-level counts are derived by dividing volume counts by four modalities "
    "(T1WI, T1C, T2WI, T2FLAIR).",
    ha="center",
    fontsize=9
)

# layout
ax.set_ylim(0, max(totals) * 1.25)

plt.tight_layout()

# save
plt.savefig("final_stacked_bar_clean.png", dpi=300, bbox_inches="tight")
plt.close()