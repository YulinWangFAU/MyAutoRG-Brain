#!/bin/bash
#SBATCH --job-name=eval_ratescore
#SBATCH --output=logs/eval_ratescore_%j.out
#SBATCH --error=logs/eval_ratescore_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -euo pipefail

export SLURM_EXPORT_ENV=ALL

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export HF_HOME=/home/woody/iwi5/iwi5325h/huggingface_cache
export TRANSFORMERS_CACHE=/home/woody/iwi5/iwi5325h/huggingface_cache
export TOKENIZERS_PARALLELISM=false

module load python/3.12-conda
conda activate eval_ratescore

cd "$HOME/MyAutoRG-Brain/report_evaluation_inputs/evaluation"

mkdir -p "$HF_HOME" results

nvidia-smi
python check_eval_env.py
python build_unified_dataset.py
python evaluate_ratescore.py \
  --input prepared/unified_case_level_reports.csv \
  --output-dir results \
  --bootstrap-iterations 10000 \
  --seed 42

echo "RaTEScore evaluation completed"
echo "Summary: $PWD/results/ratescore_summary.csv"
echo "Per-case: $PWD/results/ratescore_per_case.csv"
