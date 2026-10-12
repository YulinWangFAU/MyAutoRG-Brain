# Experiment Status: Grounded Multimodal Brain MRI Report Generation

Last updated: 2026-10-05

This document is the persistent handoff record for the thesis experiments. Read it before continuing the project in a new conversation. It records confirmed results separately from planned or incomplete work.

## 1. Research scope

The project studies multimodal brain MRI report generation using four MRI sequences (T1, T1ce/T1C, T2, and FLAIR). The main question is:

> How should information from multiple MRI modalities be fused to generate clinically faithful brain MRI reports, and how robust are the fusion methods when the pipeline uses predicted abnormality masks instead of ground-truth abnormality masks?

The evaluated method families are:

- Direct AutoRG baseline (fixed pretrained checkpoint).
- Single-modality AutoRG baselines: T1, T1C, T2, and FLAIR.
- Text-level fusion: T5-Small, Flan-T5-Large-LoRA, and Flan-T5-Large-LoRA-PullLoss.
- Feature-level fusion: Mean, Learnable Weighted Mean, Concat-Projection, and Token-Gated.
- VLM baselines: Qwen2.5-VL Zero-Shot and Few-Shot.

## 2. Important terminology and experimental conditions

### B1 / GT-mask condition

The historical label `gt_mask` or `gt_abnormal_mask` does **not** mean that every mask is ground truth:

- Anatomy mask: predicted by the AutoRG segmentation model.
- Abnormality mask: ground truth.

This is therefore an oracle-abnormality-mask condition, not a fully ground-truth pipeline.

### A-series / predicted-mask condition

- Anatomy mask: predicted by the AutoRG segmentation model.
- Abnormality mask: predicted by the AutoRG segmentation model.

This is the deployable/realistic pipeline condition.

### Fixed baseline and random seeds

- Direct AutoRG is a fixed pretrained baseline. It is evaluated once and is not assigned artificial seed replicates.
- Mean fusion has no trainable fusion parameters and does not require five retraining runs solely for the fusion module.
- Trainable feature-fusion methods should be trained with five seeds: `42`, `12345`, `2026`, `3407`, and `5901`.
- Report seed-level mean and standard deviation for trained methods. Do not duplicate the fixed baseline score five times.
- Existing train/validation/test splits remain fixed; seed experiments measure optimization variability, not cross-validation variability.

## 3. Confirmed completed experiments

All values below are raw RaTEScore values on a 0-to-1 scale. Multiply by 100 only for paper tables that use percentage presentation.

### B1: GT abnormality mask evaluation (92 test cases)

| Category | Method | Mean RaTEScore |
|---|---|---:|
| Baseline | Direct AutoRG | 0.5772 |
| Single modality | AutoRG-T1 | 0.5138 |
| Single modality | AutoRG-T1C | 0.4593 |
| Single modality | AutoRG-T2 | 0.5352 |
| Single modality | AutoRG-FLAIR | 0.4799 |
| Feature level | Mean | 0.5444 |
| Feature level | Weighted-Mean | 0.5331 |
| Feature level | Concat-Projection | 0.6249 |
| Feature level | Token-Gated | 0.6179 |
| Text level, AutoRG input | T5-Small | 0.6360 |
| Text level, AutoRG input | Flan-T5-Large-LoRA | 0.6059 |
| Text level, AutoRG input | Flan-T5-Large-LoRA-PullLoss | 0.6053 |
| Text level, oracle GT text input | T5-Small | 0.7880 |
| Text level, oracle GT text input | Flan-T5-Large-LoRA | 0.7278 |
| Text level, oracle GT text input | Flan-T5-Large-LoRA-PullLoss | 0.7215 |
| VLM | Qwen2.5-VL Few-Shot | 0.5638 |
| VLM | Qwen2.5-VL Zero-Shot | 0.4171 |

Paired B1 findings:

- Concat-Projection vs Direct AutoRG: +0.0477, 95% CI [0.0294, 0.0667], Holm p=0.00016.
- Token-Gated vs Direct AutoRG: +0.0407, 95% CI [0.0212, 0.0605], Holm p=0.00054.
- T5-Small vs Direct AutoRG: +0.0588, 95% CI [0.0453, 0.0728], Holm p=0.00016.
- Concat-Projection vs T5-Small: -0.0111, 95% CI [-0.0280, 0.0053], Holm p=0.3887; no significant difference.
- Concat-Projection and Token-Gated significantly outperform Mean and Weighted-Mean.
- Concat-Projection vs Token-Gated is not significant.
- Few-shot Qwen strongly outperforms zero-shot Qwen but remains below the strongest fusion methods.

### A1: Direct and single-modality AutoRG with predicted abnormality masks

| Category | Method | Mean RaTEScore |
|---|---|---:|
| Baseline | Direct AutoRG | 0.5802 |
| Single modality | AutoRG-T1 | 0.5186 |
| Single modality | AutoRG-T1C | 0.4576 |
| Single modality | AutoRG-T2 | 0.5273 |
| Single modality | AutoRG-FLAIR | 0.4728 |

The slightly higher Direct AutoRG score under predicted masks (0.5802 vs 0.5772) is a small empirical difference, not evidence that predicted masks are generally superior. Report the paired test/effect size rather than interpreting the raw means alone.

### A2: Text-level fusion with predicted abnormality masks

| Method | Mean RaTEScore |
|---|---:|
| T5-Small | 0.6399 |
| Flan-T5-Large-LoRA | 0.6114 |
| Flan-T5-Large-LoRA-PullLoss | 0.6124 |

Paired GT-vs-predicted-mask findings:

- T5-Small: Pred-GT = +0.0040, 95% CI [-0.0086, 0.0166], Holm p=1.
- Flan-T5-Large-LoRA: +0.0054, 95% CI [-0.0119, 0.0225], Holm p=1.
- Flan-T5-Large-LoRA-PullLoss: +0.0071, 95% CI [-0.0103, 0.0242], Holm p=1.
- No significant degradation from replacing GT abnormality masks with predicted masks for these text-level models.

### A3: Feature-level fusion with predicted abnormality masks

| Method | Mean RaTEScore |
|---|---:|
| Mean | 0.4544 |
| Weighted-Mean | 0.4201 |
| Concat-Projection | 0.6158 |
| Token-Gated | 0.6114 |

Paired GT-vs-predicted-mask findings:

- Mean: Pred-GT = -0.0900, 95% CI [-0.1173, -0.0628], Holm p=4e-05.
- Weighted-Mean: -0.1130, 95% CI [-0.1430, -0.0833], Holm p=4e-05.
- Concat-Projection: -0.0091, 95% CI [-0.0273, 0.0092], Holm p=0.6611.
- Token-Gated: -0.0065, 95% CI [-0.0218, 0.0085], Holm p=0.6611.
- Concat-Projection and Token-Gated are robust to predicted-mask input; Mean and Weighted-Mean degrade substantially.

### Predicted abnormality mask generation and quality checks

- 368 modality-level predicted reports/masks were generated: 92 cases x 4 modalities.
- Five predicted abnormality masks were empty, involving three of the 92 cases.
- Re-running those five inputs produced the same empty masks, indicating deterministic model behavior rather than a transient job failure.
- Empty masks:
  - `BraTS-MEN-00583-000-t1c_ab.nii.gz`
  - `BraTS-MEN-00583-000-t2w_ab.nii.gz`
  - `BraTS-MEN-01062-000-t1n_ab.nii.gz`
  - `BraTS-MEN-01062-000-t2w_ab.nii.gz`
  - `BraTS-MEN-01285-000-t1c_ab.nii.gz`
- A segmentation-quality evaluator was run, and its CSV outputs should be retained with the report results.
- Case-level T1C Dice and report-score change do not show a simple monotonic relationship. In particular, the correlation is near zero for Concat-Projection and Token-Gated; do not claim that lower Dice automatically causes worse report quality.

## 4. Evaluation metric conventions

### Scale

- RaTEScore and many F1-style metrics naturally return values in `[0, 1]`.
- The AutoRG-Brain paper reports several metrics multiplied by 100.
- A local value such as `0.6249` is therefore `62.49` in a percentage-style paper table; the calculation is not missing a factor of 100 if the raw-scale convention is stated.
- RadCliQ is an error score; lower is better and it should not be multiplied by 100 unless explicitly defined otherwise.

### Paper-compatible metric suite

The recommended primary metric table contains:

- BLEU-4 (higher is better; paper-style percentage).
- ROUGE-1 recall (higher is better; paper-style percentage).
- BERTScore F1 with `bert-base-uncased` (higher is better; paper-style percentage).
- RadGraph entity/relation F1 (higher is better; paper-style percentage).
- RadCliQ v1 raw error score (lower is better).
- RaTEScore (higher is better; paper-style percentage).

The newly added evaluator is:

- `report_evaluation_inputs/evaluation/evaluate_paper_metrics.py`
- `report_evaluation_inputs/evaluation/requirements_paper_metrics.txt`
- `report_evaluation_inputs/evaluation/submit_paper_metrics_v100.sbatch`

The evaluator writes:

- `paper_metrics_per_case.csv`
- `paper_metrics_summary.csv`
- `paper_metrics_methodology.json`

The evaluator was added and committed, but a full HPC run of all paper metrics must still be confirmed.

## 5. Reproducibility code already prepared

Random-seed support was added to:

- `AutoRG_Brain/train_llm_multi.py`
- `AutoRG_Brain/network_training/nnUNetTrainerV2_llm_resize_multi.py`

The five-seed feature-fusion array script is:

- `AutoRG_Brain/submit_train_feature_fusion_multiseed_v100.sbatch`

It covers 3 trainable fusion methods x 5 seeds = 15 training jobs:

- Weighted-Mean
- Concat-Projection
- Token-Gated

Before submitting, inspect checkpoint/output paths and confirm that each seed writes to a unique directory. The five-seed training jobs have not yet been recorded as completed in this document.

## 6. Key filesystem locations

### Local Mac

- Main repository: `/Users/wangyulin/LLM-MRI-THESIS-PROJECT/MyAutoRG-Brain`
- Original paper code: `/Users/wangyulin/LLM-MRI-THESIS-PROJECT/AutoRG-Brain`
- Downloaded evaluation archive: `/Users/wangyulin/LLM-MRI-THESIS-PROJECT/Task003_llm_fusion_jsons/evaluation_results_2026-09-15`
- Presentation outputs: `/Users/wangyulin/LLM-MRI-THESIS-PROJECT/ppt_outputs`

### HPC

- Repository: `~/MyAutoRG-Brain`
- Predicted-mask reports/masks: `/home/woody/iwi5/iwi5325h/autorg_output_test_predmask`
- GT-mask anatomy outputs: `/home/woody/iwi5/iwi5325h/autorg_output_test`
- Predicted-mask feature-fusion outputs: `/home/woody/iwi5/iwi5325h/feature_fusion_predmask`
- Text-fusion predicted-mask outputs: `/home/woody/iwi5/iwi5325h/text_fusion_predmask`
- Shared storage for checkpoints: `/home/woody/iwi5/iwi5325h/MyAutoRG_storage/checkpoints`
- Conda environments: `autorg`, `eval_ratescore`, and `eval_radgraph`

Common initialization:

```bash
module load python/3.12-conda
conda activate eval_ratescore
```

## 7. Current interpretation

The strongest supported conclusion is not simply that multimodal fusion always helps. Instead:

1. Multimodal approaches generally outperform single-modality AutoRG baselines.
2. Fusion design is critical: Concat-Projection and Token-Gated substantially outperform simple Mean and Weighted-Mean fusion.
3. Concat-Projection and Token-Gated retain their performance when GT abnormality masks are replaced by predicted masks.
4. Text-level fusion is also insensitive to the mask-source replacement in this test set.
5. Simple averaging methods are much more vulnerable to prediction errors in upstream masks.
6. Mask Dice alone does not fully explain report-generation quality; qualitative case-level error analysis is required.

## 8. Remaining work, in priority order

### Priority 1: Run the complete paper-compatible metric suite

Run BLEU-4, ROUGE-1, BERTScore, RadGraph, RadCliQ, and RaTEScore for the final aligned method set. Preserve per-case outputs and methodology metadata. This is required because RaTEScore alone is not sufficient for comparison with AutoRG-Brain and related report-generation work.

### Priority 2: Complete five-seed training/evaluation

Train Weighted-Mean, Concat-Projection, and Token-Gated using the five predefined seeds. Evaluate every seed on the same fixed test set and report:

- mean across seeds;
- standard deviation across seeds;
- individual seed scores;
- fixed Direct AutoRG reference shown once;
- paired case-level comparison for the selected final model where appropriate.

Three seeds may be acceptable for a preliminary conference submission, but five provide a stronger stability analysis and are the current project decision.

### Priority 3: Case-level error analysis

For selected representative cases, show side-by-side:

- Ground-truth report.
- Direct AutoRG report.
- Best feature-fusion report under GT abnormality mask.
- Same feature-fusion report under predicted abnormality mask.
- Predicted abnormality mask and GT mask visualization.
- Highlighted errors: omission, false positive, location/laterality, signal characteristics, enhancement, lesion size, edema, and mass effect.

Use both success and failure cases. Existing analysis identified 22 candidate cases; verify and refine this set rather than cherry-picking only favorable examples.

### Priority 4: Final statistical reporting

- Use paired bootstrap confidence intervals for case-level score differences.
- Use paired randomization/permutation tests with Holm correction for multiple comparisons.
- Report paired effect size (Cohen's dz) and win/loss counts.
- Keep case-level variability separate from across-seed variability.

### Priority 5: Optional additions

- Evaluate whether VLM baselines should also be run under precisely matched image/mask conditions.
- Add segmentation subgroup analysis for empty masks, low-Dice cases, tumor type (GLI/MEN), and modality-specific failures.
- If computationally feasible, include a stronger contemporary VLM baseline, but only after the core controlled comparisons are complete.

## 9. Git status warning

At the time this document was created, the local worktree contained multiple modified and untracked experiment files. They belong to ongoing work and must not be discarded with destructive Git commands. Review and commit changes in logically separated groups. Never commit model checkpoints, large generated datasets, secrets, or GitHub tokens.

## 10. Immediate next command sequence

On the HPC, after pulling the latest evaluator:

```bash
cd ~/MyAutoRG-Brain/report_evaluation_inputs/evaluation
module load python/3.12-conda
conda activate eval_ratescore
python -m pip install -r requirements_paper_metrics.txt
mkdir -p logs
sbatch submit_paper_metrics_v100.sbatch
```

Before submission, inspect `INPUT_CSV` and `OUTPUT_DIR` in the Slurm script so results are written to a new, clearly named directory rather than overwriting an earlier run.
