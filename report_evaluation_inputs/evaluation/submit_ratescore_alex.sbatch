#!/bin/bash -l
#
#SBATCH --job-name=brain_mri_ratescore
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=ratescore_%j.out
#SBATCH --error=ratescore_%j.err

set -euo pipefail
unset SLURM_EXPORT_ENV

module load python/3.12-conda
conda activate eval_radgraph

cd "$WORK/LLM-MRI-THESIS-PROJECT/MyAutoRG-Brain/report_evaluation_inputs/evaluation"

# Keep downloaded model weights on project storage rather than the small home quota.
export HF_HOME="$WORK/huggingface_cache"
export TOKENIZERS_PARALLELISM=false
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
