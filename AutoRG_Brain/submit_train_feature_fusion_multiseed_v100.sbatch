#!/bin/bash
#SBATCH --job-name=ff_multiseed
#SBATCH --output=logs/ff_multiseed_%A_%a.out
#SBATCH --error=logs/ff_multiseed_%A_%a.err
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
export AUTORG_REAL_BATCH_SIZE=1

module load python/3.12-conda
source ~/.bashrc
conda activate autorg

PROJECT_DIR=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain
OUTPUT_ROOT=/home/woody/iwi5/iwi5325h/autorg_runs_multiseed
TASK=Task003_llm_fusion
TRAIN_FILE="$PROJECT_DIR/../raw_data/Task003_llm_fusion/train_fusion_case_level.json"
PLANS_FILE="$PROJECT_DIR/utils_file/nnUNetPlansv2.1_plans_3D.pkl"
SEG_DIR="$PROJECT_DIR/checkpoints/AutoRG_Brain_SEG"
SEG_PRETRAIN="$SEG_DIR/seg_model.model"

METHODS=(weighted_mean concat_projection token_gated)
SEEDS=(42 12345 2026 3407 5901)

METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
FUSION_TYPE=${METHODS[$METHOD_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
EXPERIMENT_NAME="train_fusion_${FUSION_TYPE}_layer2_b1_train322_val46_seed${SEED}"

# PYTHONHASHSEED must be set before Python starts. The Python entry point also
# seeds Python, NumPy, PyTorch, CUDA, and the trainer with the same value.
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_ROOT" "$HF_HOME"
cd "$PROJECT_DIR"

test -f "$TRAIN_FILE" || { echo "Missing train file: $TRAIN_FILE" >&2; exit 1; }
test -f "$PLANS_FILE" || { echo "Missing plans file: $PLANS_FILE" >&2; exit 1; }

if [ ! -f "$SEG_PRETRAIN" ]; then
  echo "Missing segmentation checkpoint: $SEG_PRETRAIN" >&2
  echo "Available .model files under $SEG_DIR:" >&2
  find -L "$SEG_DIR" -type f -name '*.model' -print >&2 || true
  echo "Update SEG_PRETRAIN in this script if your filename differs." >&2
  exit 1
fi

echo "fusion_type=$FUSION_TYPE"
echo "seed=$SEED"
echo "seg_pretrained=$SEG_PRETRAIN"
echo "output=$OUTPUT_ROOT/$EXPERIMENT_NAME"
nvidia-smi

python train_llm_feature_fusion.py \
  3d_fullres nnUNetTrainerV2 "$TASK" 0 \
  --dataset six \
  --size 4 \
  --no_aug \
  --deterministic \
  --input_mode multi \
  --fusion_type "$FUSION_TYPE" \
  --fusion_gate_hidden_dim 256 \
  --fusion_dropout 0.1 \
  --feature_layer 2 \
  -train_batch 100 \
  -val_batch 20 \
  --network_type share \
  -seg_pretrained "$SEG_PRETRAIN" \
  -train "$TRAIN_FILE" \
  --plans_file "$PLANS_FILE" \
  --output_root "$OUTPUT_ROOT" \
  --experiment_name "$EXPERIMENT_NAME" \
  --seed "$SEED"
