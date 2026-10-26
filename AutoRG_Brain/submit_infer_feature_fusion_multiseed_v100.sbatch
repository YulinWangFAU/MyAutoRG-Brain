#!/bin/bash
#SBATCH --job-name=ff_ms_infer
#SBATCH --output=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/logs/ff_ms_infer_%A_%a.out
#SBATCH --error=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain/logs/ff_ms_infer_%A_%a.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
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
export AUTORG_REAL_BATCH_SIZE=1
export AUTORG_AUG_NUM_THREADS=1
export AUTORG_AUG_CACHE_PER_THREAD=1

module load python/3.12-conda
source ~/.bashrc
conda activate autorg

PROJECT_DIR=/home/hpc/iwi5/iwi5325h/MyAutoRG-Brain/AutoRG_Brain
RUN_ROOT=/home/woody/iwi5/iwi5325h/autorg_runs_multiseed
OUTPUT_ROOT=/home/woody/iwi5/iwi5325h/feature_fusion_predictions_multiseed
PLANS="$PROJECT_DIR/utils_file/nnUNetPlansv2.1_plans_3D.pkl"

METHODS=(weighted_mean concat_projection token_gated)
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

EXPERIMENT="train_fusion_${METHOD}_layer2_b1_train322_val46_seed${SEED}"
CHECKPOINT="$RUN_ROOT/$EXPERIMENT/fold_0/model_best_rouge.model"

if [ "$INPUT_SOURCE" = "gt_abnormal_mask" ]; then
  TASK=Task003_llm_fusion
  TEST_JSON="$PROJECT_DIR/../raw_data/Task003_llm_fusion/fusion_case_level_train_val_test_abnormal.json"
else
  TASK=Task004_llm_fusion_predmask
  TEST_JSON="$PROJECT_DIR/../raw_data/Task004_llm_fusion_predmask/fusion_predmask_inference.json"
fi

OUTPUT_DIR="$OUTPUT_ROOT/$INPUT_SOURCE/${METHOD}_seed${SEED}"
OUTPUT_PATH="$OUTPUT_DIR/pred_test.json"

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR" "$HF_HOME"
cd "$PROJECT_DIR"

test -f "$PLANS" || { echo "Missing plans: $PLANS" >&2; exit 1; }
test -f "$TEST_JSON" || { echo "Missing test JSON: $TEST_JSON" >&2; exit 1; }
test -f "$CHECKPOINT" || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
test -f "${CHECKPOINT}.pkl" || { echo "Missing checkpoint metadata: ${CHECKPOINT}.pkl" >&2; exit 1; }
if [ -e "$OUTPUT_PATH" ]; then
  echo "Refusing to overwrite existing output: $OUTPUT_PATH" >&2
  exit 2
fi

python - "$TEST_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
cases = payload.get("test")
reports = payload.get("region_report", {}).get("test")
if not isinstance(cases, list) or len(cases) != 92:
    raise ValueError(f"{path}: expected 92 test cases")
if not isinstance(reports, dict) or set(cases) != set(reports):
    raise ValueError(f"{path}: test cases and region_report/test do not align")
print(f"Validated test JSON: {path} ({len(cases)} cases)")
PY

echo "array_task=$SLURM_ARRAY_TASK_ID"
echo "fusion_type=$METHOD"
echo "seed=$SEED"
echo "input_source=$INPUT_SOURCE"
echo "task=$TASK"
echo "checkpoint=$CHECKPOINT"
echo "output=$OUTPUT_PATH"
nvidia-smi

python infer_feature_fusion.py \
  3d_fullres nnUNetTrainerV2 "$TASK" 0 \
  -train "$TEST_JSON" \
  --plans_file "$PLANS" \
  --checkpoint "$CHECKPOINT" \
  --dataset six \
  --size 4 \
  --feature_layer 2 \
  --network_type share \
  --input_mode multi \
  --keep_compressed \
  --fusion_type "$METHOD" \
  --split test \
  --output_folder "$OUTPUT_DIR" \
  --output_name pred_test.json

echo "Feature-fusion inference completed: $OUTPUT_PATH"
