#!/bin/bash
#SBATCH --job-name=ratescore_stats
#SBATCH --output=logs/ratescore_stats_%j.out
#SBATCH --error=logs/ratescore_stats_%j.err
#SBATCH --partition=v100
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

module load python/3.12-conda
source ~/.bashrc
conda activate eval_ratescore

cd "$HOME/MyAutoRG-Brain/report_evaluation_inputs/evaluation"
mkdir -p logs results_b1_gtmask/paired_statistics

python paired_ratescore_analysis.py \
  --input results_b1_gtmask/ratescore_per_case.csv \
  --output-dir results_b1_gtmask/paired_statistics \
  --bootstrap-iterations 10000 \
  --permutation-iterations 100000 \
  --seed 42

echo "Paired RaTEScore analysis completed"
