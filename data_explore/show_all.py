import os
import nibabel as nib
import matplotlib

# 解决 PyCharm Backend 报错
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 路径建议使用 raw string 避免转义字符问题
base_path = r"/Users/wangyulin/LLM-MRI-THESIS-PROJECT/BraTS 2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00121-000"
# base_path = r"/Users/wangyulin/LLM-MRI-THESIS-PROJECT/BraTS 2023/BraTS-MEN-Train/BraTS-MEN-00017-000"
modalities = ['t1n', 't1c', 't2w', 't2f', 'seg']
titles = ['T1 Native', 'T1c (Gd)', 'T2 Weighted', 'T2 FLAIR', 'Segmentation']


def visualize_brats_slice(data_path, slice_idx=None):
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # 获取病人ID (文件夹名)
    patient_id = os.path.basename(data_path)

    for i, mod in enumerate(modalities):
        file_name = f"{patient_id}-{mod}.nii.gz"
        full_path = os.path.join(data_path, file_name)

        if not os.path.exists(full_path):
            print(f"警告: 找不到文件 {full_path}")
            continue

        # 加载数据
        img_obj = nib.load(full_path)
        img_data = img_obj.get_fdata()

        # 默认取中间层
        if slice_idx is None:
            slice_idx = img_data.shape[2] // 2

        # 提取并旋转切片
        display_data = np.rot90(img_data[:, :, slice_idx])

        # 绘图
        ax = axes[i]
        if mod == 'seg':
            # 分割图用 jet 颜色增加对比度
            ax.imshow(display_data, cmap='nipy_spectral')
        else:
            ax.imshow(display_data, cmap='gray')

        ax.set_title(f"{titles[i]}\nSlice: {slice_idx}")
        ax.axis('off')

    plt.suptitle(f"Patient: {patient_id}", fontsize=16)
    plt.tight_layout()

    # 尝试显示
    try:
        plt.show()
    except Exception as e:
        print(f"无法直接显示图像，正在保存到本地: {e}")
        plt.savefig("debug_view.png")


def visualize_with_overlay(data_path, slice_idx=77):
    # 加载 T1c 作为背景
    t1c = nib.load(os.path.join(data_path, f"{os.path.basename(data_path)}-t1c.nii.gz")).get_fdata()
    # 加载 Seg 作为遮罩
    seg = nib.load(os.path.join(data_path, f"{os.path.basename(data_path)}-seg.nii.gz")).get_fdata()

    background = np.rot90(t1c[:, :, slice_idx])
    mask = np.rot90(seg[:, :, slice_idx])

    plt.figure(figsize=(8, 8))
    plt.imshow(background, cmap='gray')  # 底图用灰度

    # 将 mask 中为 0 的部分设为透明 (NaN)
    mask_display = np.ma.masked_where(mask == 0, mask)

    plt.imshow(mask_display, cmap='autumn', alpha=0.5)  # 遮罩用暖色，透明度 0.5
    plt.title("T1c with Segmentation Overlay")
    plt.axis('off')
    plt.show()


def show_subregions(data_path, slice_idx=77):
    # 加载分割文件
    seg_img = nib.load(os.path.join(data_path, f"{os.path.basename(data_path)}-seg.nii.gz")).get_fdata()
    seg_slice = np.rot90(seg_img[:, :, slice_idx])

    # 逻辑提取子区域
    wt = (seg_slice > 0).astype(int)  # 全肿瘤 (1, 2, 3)
    tc = ((seg_slice == 1) | (seg_slice == 3)).astype(int)  # 核心 (1, 3)
    et = (seg_slice == 3).astype(int)  # 仅增强 (3)

    regions = [wt, tc, et]
    names = ['Whole Tumor (WT)', 'Tumor Core (TC)', 'Enhancing Tumor (ET)']
    colors = ['Purples', 'Greens', 'Reds']

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(regions[i], cmap=colors[i])
        plt.title(names[i])
        plt.axis('off')
    plt.show()


def visualize_best_slice(data_path):
    # 加载分割图
    patient_id = os.path.basename(data_path)
    seg_path = os.path.join(data_path, f"{patient_id}-seg.nii.gz")
    seg_data = nib.load(seg_path).get_fdata()

    # 【核心逻辑】计算每一层切片的非零像素数量，找到肿瘤最明显的一层
    z_counts = np.sum(seg_data > 0, axis=(0, 1))
    best_slice_idx = np.argmax(z_counts)

    if z_counts[best_slice_idx] == 0:
        print(f"警告：该病例 {patient_id} 的全卷中似乎没有任何分割标签！")
        return

    print(f"自动选择肿瘤面积最大的切片: 第 {best_slice_idx} 层")

    # 使用之前的绘图逻辑，但传入 best_slice_idx
    visualize_brats_slice(data_path, slice_idx=best_slice_idx)


# 调用
# visualize_best_slice(base_path)
if __name__ == "__main__":
    visualize_brats_slice(base_path)
    # visualize_with_overlay(base_path)
    # show_subregions(base_path)
    # visualize_best_slice(base_path)