#!/bin/bash -l
#
#SBATCH --job-name=autorg_train_llm
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

unset SLURM_EXPORT_ENV

module load python
conda activate autorg-brain

cd "$WORK/LLM-MRI-THESIS-PROJECT/MyAutoRG-Brain/AutoRG_Brain"
mkdir -p logs

python ../hpc/check_gpu_env.py

python train_llm_multi.py \
  3d_fullres nnUNetTrainerV2 002 0 \
  --dataset six \
  --size 4 \
  --no_aug \
  --input_mode multi \
  --fusion_type token_gated \
  --fusion_gate_hidden_dim 256 \
  --fusion_dropout 0.1 \
  --feature_layer 2 \
  -train_batch 10 \
  -val_batch 10 \
  --network_type share \
  -seg_pretrained "$WORK/checkpoints/AutoRG_Brain/AutoRG_Brain_SEG.model" \
  -train "$WORK/data/autorg/Task002_llm_test/test_file.json" \
  --plans_file utils_file/nnUNetPlansv2.1_plans_3D.pkl
