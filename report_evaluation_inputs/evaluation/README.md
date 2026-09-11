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
