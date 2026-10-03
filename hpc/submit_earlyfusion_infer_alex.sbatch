#!/bin/bash -l
#
#SBATCH --job-name=autorg_earlyfusion
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

unset SLURM_EXPORT_ENV

module load python
conda activate autorg-brain

cd "$WORK/LLM-MRI-THESIS-PROJECT/MyAutoRG-Brain/AutoRG_Brain"
mkdir -p logs

python ../hpc/check_gpu_env.py

export AUTORG_FEATURE_DIR="$WORK/autorg_outputs/earlyfusion_features/${SLURM_JOB_ID}"
mkdir -p "$AUTORG_FEATURE_DIR"

python test_earlyfusion.py \
  --eval_mode given_mask \
  --llm_folder "$WORK/checkpoints/AutoRG_Brain" \
  --llm_chk AutoRG_Brain_RGv2 \
  --seg_folder "$WORK/checkpoints/AutoRG_Brain" \
  --seg_chk AutoRG_Brain_SEG \
  -test "$WORK/data/autorg/test_file.json" \
  -o "$WORK/autorg_outputs/earlyfusion_${SLURM_JOB_ID}" \
  --save_intermediate \
  --feature_dir "$AUTORG_FEATURE_DIR"
