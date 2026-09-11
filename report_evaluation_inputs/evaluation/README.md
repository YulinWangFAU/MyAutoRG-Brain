# Evaluation pipeline

Build and validate the unified 92-case evaluation dataset:

```bash
python build_unified_dataset.py
```

This creates:

- `prepared/unified_case_level_reports.csv`
- `prepared/unified_case_level_reports.jsonl`

The table contains 12 model/input configurations and 1,104 rows. Oracle GT-input text-fusion results are retained through the `input_source` column and must be reported separately from realistic AutoRG-input results.

Install RaTEScore dependencies and evaluate:

```bash
python -m pip install -r requirements.txt
python evaluate_ratescore.py
```

The first RaTEScore run downloads model weights. The output files are:

- `results/ratescore_per_case.csv`
- `results/ratescore_summary.csv`

The single-modal 368-record file is intentionally excluded because its evaluation unit is modality-level rather than case-level.

## Alex GPU job

After pulling the repository on Alex, first inspect the package without running a model:

```bash
cd "$HOME/MyAutoRG-Brain/report_evaluation_inputs/evaluation"
find .. -maxdepth 3 -type f | sort
wc -l prepared/unified_case_level_reports.csv
python build_unified_dataset.py
```

Load the existing environment and inspect dependencies on the login node:

```bash
module load python/3.12-conda
conda activate eval_ratescore
python --version
python -m pip check
python -m pip install -r requirements.txt
```

Do not run `check_eval_env.py` on the login node because it intentionally requires CUDA. Submit the GPU evaluation with:

```bash
mkdir -p logs
sbatch submit_ratescore_alex.sbatch
squeue --me
```

Inspect the logs after the job starts:

```bash
tail -f logs/eval_ratescore_<job-id>.out
tail -f logs/eval_ratescore_<job-id>.err
```
