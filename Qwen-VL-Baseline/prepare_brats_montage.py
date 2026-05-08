# -*- coding: utf-8 -*-
"""
Created on 2026/5/7 18:15

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw
from tqdm import tqdm

MODALITIES = {
    "t1n": "T1",
    "t1c": "T1ce",
    "t2w": "T2",
    "t2f": "FLAIR",
}


def normalize_slice(x):
    x = np.nan_to_num(x)

    if np.any(x > 0):
        p1, p99 = np.percentile(x[x > 0], [1, 99])
    else:
        p1, p99 = x.min(), x.max()

    x = np.clip(x, p1, p99)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    return (x * 255).astype(np.uint8)


def load_nii(path):
    return nib.load(path).get_fdata()


def find_largest_tumor_slice(seg):
    mask_area = np.sum(seg > 0, axis=(0, 1))
    return int(np.argmax(mask_area))


def make_panel(case_dir, out_path, image_size=384):
    case_id = os.path.basename(case_dir)

    print("\n==============================")
    print("Processing case:", case_id)
    print("Saving to:", out_path)

    seg_path = os.path.join(case_dir, f"{case_id}-seg.nii.gz")

    if not os.path.exists(seg_path):
        print("Missing segmentation file:", seg_path)
        return

    seg = load_nii(seg_path)

    z = find_largest_tumor_slice(seg)
    print("Largest tumor slice z =", z)

    panels = []

    for suffix, label in MODALITIES.items():
        nii_path = os.path.join(case_dir, f"{case_id}-{suffix}.nii.gz")

        if not os.path.exists(nii_path):
            print("Missing modality file:", nii_path)
            return

        print("Loading:", nii_path)

        data = load_nii(nii_path)
        img = normalize_slice(data[:, :, z])
        img = np.rot90(img)

        pil_img = Image.fromarray(img).convert("RGB")
        pil_img = pil_img.resize((image_size, image_size))

        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([0, 0, 120, 32], fill=(0, 0, 0))
        draw.text((10, 8), label, fill=(255, 255, 255))

        panels.append(pil_img)

    canvas = Image.new(
        "RGB",
        (image_size * 2, image_size * 2),
        "black"
    )

    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (image_size, 0))
    canvas.paste(panels[2], (0, image_size))
    canvas.paste(panels[3], (image_size, image_size))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

    print("Saved successfully:", out_path)


def main():
    split_dir = "/home/woody/iwi5/iwi5325h/BraTS_filtered/train"
    out_dir = "/home/woody/iwi5/iwi5325h/qwen_mri_montage/train"

    print("Input dir :", split_dir)
    print("Output dir:", out_dir)

    case_ids = sorted([
        name for name in os.listdir(split_dir)
        if not name.startswith("._")
        and os.path.isdir(os.path.join(split_dir, name))
    ])

    # 先只测试前 5 个真实 case
    case_ids = case_ids[:5]

    print("Found", len(case_ids), "real cases")

    for case_id in tqdm(case_ids):
        case_dir = os.path.join(split_dir, case_id)

        print("\nChecking:", case_dir)
        print("isdir =", os.path.isdir(case_dir))

        out_path = os.path.join(out_dir, f"{case_id}.png")

        print("out_path =", out_path)

        make_panel(case_dir, out_path)


if __name__ == "__main__":
    main()