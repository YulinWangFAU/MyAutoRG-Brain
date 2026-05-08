# -*- coding: utf-8 -*-
"""
Created on 2026/5/7 18:15

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import csv
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
    z = int(np.argmax(mask_area))
    max_area = int(mask_area[z])
    return z, max_area


def make_panel(case_dir, out_path, split_name, image_size=384):
    case_id = os.path.basename(case_dir)

    print("\n==============================")
    print("Split:", split_name)
    print("Processing case:", case_id)
    print("Input case dir:", case_dir)
    print("Saving to:", out_path)

    seg_path = os.path.join(case_dir, f"{case_id}-seg.nii.gz")

    if not os.path.exists(seg_path):
        print("Missing segmentation file:", seg_path)
        return None

    seg = load_nii(seg_path)

    z, max_tumor_area = find_largest_tumor_slice(seg)

    print("Largest tumor slice z =", z)
    print("Max tumor area =", max_tumor_area)

    panels = []
    modality_paths = {}

    for suffix, label in MODALITIES.items():
        nii_path = os.path.join(case_dir, f"{case_id}-{suffix}.nii.gz")
        modality_paths[suffix] = nii_path

        if not os.path.exists(nii_path):
            print("Missing modality file:", nii_path)
            return None

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

    info = {
        "split": split_name,
        "case_id": case_id,
        "case_dir": case_dir,
        "seg_path": seg_path,
        "t1n_path": modality_paths["t1n"],
        "t1c_path": modality_paths["t1c"],
        "t2w_path": modality_paths["t2w"],
        "t2f_path": modality_paths["t2f"],
        "largest_tumor_slice_z": z,
        "max_tumor_area_pixels": max_tumor_area,
        "output_png_path": out_path,
    }

    return info


def get_case_ids(split_dir):
    case_ids = sorted([
        name for name in os.listdir(split_dir)
        if not name.startswith("._")
        and os.path.isdir(os.path.join(split_dir, name))
    ])
    return case_ids


def process_split(split_name, input_root, output_root, max_cases=None):
    split_dir = os.path.join(input_root, split_name)
    out_dir = os.path.join(output_root, split_name)

    print("\n################################")
    print("Processing split:", split_name)
    print("Input dir :", split_dir)
    print("Output dir:", out_dir)

    if not os.path.exists(split_dir):
        print("Split dir does not exist:", split_dir)
        return []

    case_ids = get_case_ids(split_dir)

    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    print("Found", len(case_ids), "real cases")

    split_records = []

    for case_id in tqdm(case_ids):
        case_dir = os.path.join(split_dir, case_id)
        out_path = os.path.join(out_dir, f"{case_id}.png")

        record = make_panel(
            case_dir=case_dir,
            out_path=out_path,
            split_name=split_name,
        )

        if record is not None:
            split_records.append(record)

    return split_records


def save_metadata_csv(records, csv_path):
    if len(records) == 0:
        print("No records to save.")
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "split",
        "case_id",
        "case_dir",
        "seg_path",
        "t1n_path",
        "t1c_path",
        "t2w_path",
        "t2f_path",
        "largest_tumor_slice_z",
        "max_tumor_area_pixels",
        "output_png_path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\nMetadata CSV saved to:", csv_path)


def main():
    input_root = "/home/woody/iwi5/iwi5325h/BraTS_filtered"
    output_root = "/home/woody/iwi5/iwi5325h/qwen_mri_montage"

    splits = ["train", "val", "test"]

    all_records = []

    for split_name in splits:
        records = process_split(
            split_name=split_name,
            input_root=input_root,
            output_root=output_root,
            max_cases=None,   # 如果只想测试前5个，改成 max_cases=5
        )
        all_records.extend(records)


    csv_path = os.path.join(output_root, "brats_montage_metadata.csv")
    save_metadata_csv(all_records, csv_path)

    print("\nAll done.")
    print("Total generated cases:", len(all_records))


if __name__ == "__main__":
    main()