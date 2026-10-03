# -*- coding: utf-8 -*-
"""
Created on 2026/5/5 00:04

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import matplotlib.pyplot as plt

# -----------------------------
# 数据
# -----------------------------
splits = ["Training", "Validation", "Test"]

# image-level
image_counts = [644, 92, 184]

# 每个split再分成 GLI / MEN（各一半）
outer_sizes = []
inner_labels = []

for split, val in zip(splits, image_counts):
    outer_sizes.extend([val/2, val/2])  # GLI + MEN
    inner_labels.extend([f"{split}-GLI", f"{split}-MEN"])

# -----------------------------
# 颜色设计（重点🔥）
# -----------------------------
# 黄色系（GLI）
yellow = ["#F7E6A6", "#F4D06F", "#FFF3C4"]

# 紫色系（MEN）
purple = ["#D8C4F0", "#CDB4DB", "#EADCF8"]

# 外层颜色（交替）
outer_colors = []
for y, p in zip(yellow, purple):
    outer_colors.extend([y, p])

# 内层颜色（split）
inner_colors = ["#F5F5F5"] * 3  # 浅灰

# -----------------------------
# 画图
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))

# 外层（GLI + MEN）
ax.pie(
    outer_sizes,
    radius=1,
    labels=inner_labels,
    labeldistance=1.05,
    colors=outer_colors,
    wedgeprops=dict(width=0.3, edgecolor='white')
)

# 内层（Training / Val / Test）
ax.pie(
    image_counts,
    radius=0.7,
    labels=splits,
    labeldistance=0.6,
    colors=inner_colors,
    wedgeprops=dict(width=0.3, edgecolor='white')
)

ax.set_title("Dataset Split with Subdataset Distribution", fontsize=14)

plt.savefig("nested_pie_dataset.png", dpi=300, bbox_inches="tight")
plt.close()
