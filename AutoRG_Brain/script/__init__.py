# -*- coding: utf-8 -*-
"""
Created on 2026/2/14 11:19

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
import nibabel as nib

img = nib.load("/Users/wangyulin/LLM-MRI-THESIS-PROJECT/BraTS 2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz")
print(img.header.get_zooms())
