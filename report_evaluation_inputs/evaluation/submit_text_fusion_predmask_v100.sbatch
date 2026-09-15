#!/bin/bash
#SBATCH --job-name=text_pred
#SBATCH --output=logs/text_pred_%A_%a.out
#SBATCH --error=logs/text_pred_%A_%a.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --export=NONE

set -euo pipefail

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME=/home/woody/iwi5/iwi5325h/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

module load python/3.12-conda
source ~/.bashrc
conda activate autorg

cd /home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/report_evaluation_inputs/evaluation
mkdir -p "$HF_HOME" /home/woody/iwi5/iwi5325h/text_fusion_predmask

models=(t5-small flan-lora flan-pullloss)
model="${models[$SLURM_ARRAY_TASK_ID]}"

echo "Running $model on $(hostname)"
nvidia-smi
python infer_predmask_text_fusion.py \
  --model "$model" \
  --input prepared/late_fusion_test_predmask.json \
  --output-dir /home/woody/iwi5/iwi5325h/text_fusion_predmask

echo "Completed $model"
