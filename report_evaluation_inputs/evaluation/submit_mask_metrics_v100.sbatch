#!/bin/bash
#SBATCH --job-name=mask_metrics
#SBATCH --output=logs/mask_metrics_%j.out
#SBATCH --error=logs/mask_metrics_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

module load python/3.12-conda
source ~/.bashrc
conda activate autorg

cd "$HOME/MyAutoRG-Brain/report_evaluation_inputs/evaluation"
mkdir -p logs results_a5_mask_metrics

python evaluate_predicted_masks.py \
  --prediction-dir /home/woody/iwi5/iwi5325h/autorg_output_test_predmask \
  --brats-test-root /home/woody/iwi5/iwi5325h/BraTS_filtered/test \
  --output-dir results_a5_mask_metrics \
  --bootstrap-iterations 10000 \
  --seed 42

echo "Predicted-mask segmentation evaluation completed"
