#!/bin/bash
#SBATCH --job-name=pp_pred_ff
#SBATCH --output=logs/pp_pred_ff_%j.out
#SBATCH --error=logs/pp_pred_ff_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --export=NONE

set -euo pipefail

module load python/3.12-conda
source ~/.bashrc
conda activate autorg

cd /home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain

python script/preprocess_feature_fusion_test_only.py \
  --dataset_json ../raw_data/nnUNet_raw_data/Task004_llm_fusion_predmask/dataset.json \
  --plans_file utils_file/nnUNetPlansv2.1_plans_3D.pkl \
  --preprocessed_task_dir ../preprocessed_data/Task004_llm_fusion_predmask \
  --cropped_tmp_dir ../raw_data/nnUNet_cropped_data/Task004_llm_fusion_predmask \
  --threads 4 \
  --overwrite_cropped \
  --overwrite_preprocessed

echo "A3 predicted-mask preprocessing completed"
