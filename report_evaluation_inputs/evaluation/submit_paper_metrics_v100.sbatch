#!/bin/bash
#SBATCH --job-name=paper_metrics
#SBATCH --output=logs/paper_metrics_%j.out
#SBATCH --error=logs/paper_metrics_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

export SLURM_EXPORT_ENV=ALL
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HF_HOME=/home/woody/iwi5/iwi5325h/huggingface_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export NLTK_ALLOW_PROXIED_URLOPEN=1

module load python/3.12-conda
source ~/.bashrc
conda activate eval_ratescore

EVAL_DIR="$HOME/MyAutoRG-Brain/report_evaluation_inputs/evaluation"
cd "$EVAL_DIR"
INPUT_CSV=${INPUT_CSV:-prepared/unified_case_level_reports.csv}
OUTPUT_DIR=${OUTPUT_DIR:-results_paper_metrics}
mkdir -p logs "$HF_HOME" "$OUTPUT_DIR"

nvidia-smi
python - <<'PY'
import importlib.metadata
import torch

for package in ("nltk", "radeval", "torch", "transformers"):
    print(package, importlib.metadata.version(package))
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY

python evaluate_paper_metrics.py \
  --input "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --batch-size 16 \
  --radgraph-model modern-radgraph-xl \
  --bootstrap-iterations 10000 \
  --seed 42

echo "Paper metrics evaluation completed"
echo "Summary: $PWD/$OUTPUT_DIR/paper_metrics_summary.csv"
echo "Per-case: $PWD/$OUTPUT_DIR/paper_metrics_per_case.csv"
echo "Methodology: $PWD/$OUTPUT_DIR/paper_metrics_methodology.json"
