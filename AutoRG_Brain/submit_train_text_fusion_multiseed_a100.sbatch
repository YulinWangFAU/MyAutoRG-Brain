#!/bin/bash
#SBATCH --job-name=text_multiseed
#SBATCH --output=logs/text_multiseed_%A_%a.out
#SBATCH --error=logs/text_multiseed_%A_%a.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME=/home/woody/iwi5/iwi5325h/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

module load python/3.12-conda
source ~/.bashrc
conda activate autorg

PROJECT_DIR=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain
OUTPUT_ROOT=/home/woody/iwi5/iwi5325h/text_fusion_runs_multiseed

METHODS=(t5_small flan_t5_large_lora flan_t5_large_lora_pullloss)
SCRIPTS=(
  multi_late_fusion/train_t5_late_fusion_02.py
  multi_late_fusion/train_flan_t5_large_LoRA_02.py
  multi_late_fusion/train_flan_t5_large_LoRA_PullLoss_02.py
)
SEEDS=(42 12345 2026 3407 5901)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
METHOD=${METHODS[$METHOD_INDEX]}
SCRIPT=${SCRIPTS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
OUTPUT_DIR="$OUTPUT_ROOT/${METHOD}_seed${SEED}"

export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_ROOT" "$HF_HOME"
cd "$PROJECT_DIR"

test -f "$SCRIPT" || { echo "Missing training script: $SCRIPT" >&2; exit 1; }

if [ -e "$OUTPUT_DIR" ]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  echo "Inspect or move that directory before retrying this array task." >&2
  exit 1
fi

echo "method=$METHOD"
echo "seed=$SEED"
echo "script=$SCRIPT"
echo "output=$OUTPUT_DIR"
nvidia-smi

python "$SCRIPT" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR"
