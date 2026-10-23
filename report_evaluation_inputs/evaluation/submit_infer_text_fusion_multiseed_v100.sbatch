#!/bin/bash
#SBATCH --job-name=text_ms_infer
#SBATCH --output=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/report_evaluation_inputs/evaluation/logs/text_ms_infer_%A_%a.out
#SBATCH --error=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/report_evaluation_inputs/evaluation/logs/text_ms_infer_%A_%a.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-29%2
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

PROJECT_ROOT=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain
EVALUATION_DIR="$PROJECT_ROOT/report_evaluation_inputs/evaluation"
INFERENCE_SCRIPT="$EVALUATION_DIR/infer_text_fusion_multiseed.py"

METHODS=(t5_small flan_t5_large_lora flan_t5_large_lora_pullloss)
SEEDS=(42 12345 2026 3407 5901)
INPUT_SOURCES=(gt_abnormal_mask predicted_abnormal_mask)

CONFIGS_PER_SOURCE=$((${#METHODS[@]} * ${#SEEDS[@]}))
SOURCE_INDEX=$((SLURM_ARRAY_TASK_ID / CONFIGS_PER_SOURCE))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % CONFIGS_PER_SOURCE))
METHOD_INDEX=$((CONFIG_INDEX / ${#SEEDS[@]}))
SEED_INDEX=$((CONFIG_INDEX % ${#SEEDS[@]}))

METHOD=${METHODS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
INPUT_SOURCE=${INPUT_SOURCES[$SOURCE_INDEX]}

mkdir -p "$EVALUATION_DIR/logs" "$HF_HOME"
cd "$EVALUATION_DIR"

test -f "$INFERENCE_SCRIPT" || {
  echo "Missing inference script: $INFERENCE_SCRIPT" >&2
  exit 1
}

echo "array_task=$SLURM_ARRAY_TASK_ID"
echo "method=$METHOD"
echo "seed=$SEED"
echo "input_source=$INPUT_SOURCE"
nvidia-smi

python "$INFERENCE_SCRIPT" \
  --model "$METHOD" \
  --seed "$SEED" \
  --input-source "$INPUT_SOURCE"
