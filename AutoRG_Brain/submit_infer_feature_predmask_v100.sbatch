#!/bin/bash
#SBATCH --job-name=infer_pred_ff
#SBATCH --output=logs/infer_pred_ff_%A_%a.out
#SBATCH --error=logs/infer_pred_ff_%A_%a.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --export=NONE

set -euo pipefail

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME=/home/woody/iwi5/iwi5325h/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export AUTORG_REAL_BATCH_SIZE=1
export AUTORG_AUG_NUM_THREADS=1
export AUTORG_AUG_CACHE_PER_THREAD=1

module load python/3.12-conda
source ~/.bashrc
conda activate autorg

cd /home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain

run_root=/home/woody/iwi5/iwi5325h/autorg_runs
output_root=/home/woody/iwi5/iwi5325h/feature_fusion_predmask
train_json=../raw_data/Task004_llm_fusion_predmask/fusion_predmask_inference.json
plans=utils_file/nnUNetPlansv2.1_plans_3D.pkl

fusions=(mean weighted_mean concat_projection token_gated)
experiments=(
  train_fusion_mean_layer2_b1_train322_val46
  train_fusion_weighted_mean_layer2_b1_train322_val46
  train_fusion_concat_projection_layer2_b1_train322_val46
  train_fusion_token_gated_layer2_b1_train322_val46
)

fusion="${fusions[$SLURM_ARRAY_TASK_ID]}"
experiment="${experiments[$SLURM_ARRAY_TASK_ID]}"
checkpoint="$run_root/$experiment/fold_0/model_best_rouge.model"
output_dir="$output_root/$fusion"

test -f "$checkpoint"
test -f "${checkpoint}.pkl"
mkdir -p "$output_dir"

echo "Fusion: $fusion"
echo "Checkpoint: $checkpoint"
echo "Output: $output_dir/pred_test.json"
nvidia-smi

python infer_feature_fusion.py \
  3d_fullres nnUNetTrainerV2 004 0 \
  -train "$train_json" \
  --plans_file "$plans" \
  --checkpoint "$checkpoint" \
  --dataset six \
  --size 4 \
  --feature_layer 2 \
  --network_type share \
  --input_mode multi \
  --fusion_type "$fusion" \
  --split test \
  --output_folder "$output_dir" \
  --output_name pred_test.json

echo "A3 inference completed: $fusion"
