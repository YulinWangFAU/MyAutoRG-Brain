# -*- coding: utf-8 -*-
"""
Created on 2026/5/7 18:15

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

MODALITIES = {
    "t1n": "T1",
    "t1c": "T1ce",
    "t2w": "T2",
    "t2f": "FLAIR",
}

def normalize_slice(x):
    x = np.nan_to_num(x)
    p1, p99 = np.percentile(x[x > 0], [1, 99]) if np.any(x > 0) else (x.min(), x.max())
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

    seg_path = os.path.join(case_dir, f"{case_id}-seg.nii.gz")
    seg = load_nii(seg_path)
    z = find_largest_tumor_slice(seg)

    panels = []

    for suffix, label in MODALITIES.items():
        nii_path = os.path.join(case_dir, f"{case_id}-{suffix}.nii.gz")
        data = load_nii(nii_path)

        img = normalize_slice(data[:, :, z])
        img = np.rot90(img)

        pil_img = Image.fromarray(img).convert("RGB")
        pil_img = pil_img.resize((image_size, image_size))

        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([0, 0, 120, 32], fill=(0, 0, 0))
        draw.text((10, 8), label, fill=(255, 255, 255))

        panels.append(pil_img)

    canvas = Image.new("RGB", (image_size * 2, image_size * 2), "black")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (image_size, 0))
    canvas.paste(panels[2], (0, image_size))
    canvas.paste(panels[3], (image_size, image_size))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)

def main():
    split_dir = "/home/woody/iwi5/iwi5325h/BraTS_filtered/train"
    out_dir = "/home/woody/iwi5/iwi5325h/qwen_mri_montage/train"

    case_ids = sorted(os.listdir(split_dir))[:5]

    for case_id in tqdm(case_ids):
        case_dir = os.path.join(split_dir, case_id)
        if os.path.isdir(case_dir):
            out_path = os.path.join(out_dir, f"{case_id}.png")
            make_panel(case_dir, out_path)

if __name__ == "__main__":
    main()