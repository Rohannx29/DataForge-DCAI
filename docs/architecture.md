# System Architecture & Experimental Design Rationale

## 1. Core Principle: The Controlled Experiment

This project's central claim is that **data quality improvements drive larger
performance gains than model changes, for a fixed architecture**. For that
claim to be defensible, the experimental design must isolate data quality as
the *only* independent variable.

**What is held constant across all four conditions:**
- Model architecture (`configs/model_config.yaml` — never edited per-condition)
- Random seed (`configs/base_config.yaml` -> `project.seed`)
- Training hyperparameters (learning rate, epochs, optimizer, batch size)
- Train/val/test split methodology (stratified, same seed)
- Evaluation metrics and thresholds

**What varies across conditions:**
- Which manifest CSV is fed into `src/models/train.py` (produced by different
  stages of `src/data/`, `src/labels/`, and `src/dcai/`)

## 2. The Four Experimental Conditions

| Condition | Manifest Source | What It Tests |
|---|---|---|
| `baseline` | Raw manifest, no cleaning | Lower-bound performance on unmodified data |
| `cleaned` | `src/data/cleaning.py` output | Effect of removing corrupt/duplicate files only |
| `noise_corrected` | `src/labels/noise_correction.py` output | Additional effect of fixing/removing mislabeled samples |
| `dcai_improved` | `src/dcai/` outputs + active-learning-selected labels | Combined effect of targeted augmentation, class balancing, and efficient label acquisition |

Each condition should be run `n_runs` times (default 5, see `model_config.yaml`)
with different weight initializations (but the SAME data split) to compute
mean ± 95% confidence intervals — a single run per condition is not sufficient
to claim a difference is meaningful, since training is stochastic.

## 3. Data Flow

```
Raw Data (data/raw/)
    │
    ▼
[src/data/validation.py]  → ValidationReport (corrupt files, duplicates)
    │
    ▼
[src/data/cleaning.py]    → manifest_cleaned.csv  ────────► CONDITION: cleaned
    │
    ▼
[src/labels/quality_audit.py] → LabelQualityReport
    │
    ▼
[src/labels/noise_detection.py] → flagged issue indices (via cleanlab)
    │
    ▼
[src/labels/noise_correction.py] → manifest_noise_corrected.csv ─► CONDITION: noise_corrected
    │
    ▼
[src/dcai/class_balance.py]   → imbalance diagnostics
[src/dcai/augmentation.py]    → targeted augmentation for minority class
[src/active_learning/loop.py] → efficient label acquisition
    │
    ▼
manifest_dcai_improved.csv ─────────────────────────────────────► CONDITION: dcai_improved
```

All four manifests are independently fed into the SAME `scripts/train_baseline.py`
entry point via the `--manifest` and `--condition` arguments.

## 4. Why Separate Data-Layer Concerns?

A common mistake in student DCAI projects is conflating "data cleaning,"
"label correction," and "data augmentation" into one undifferentiated
preprocessing script. This repository deliberately separates them because
they address **different failure modes** and should be evaluated
**independently** in the results section:

- `src/data/` → *technical* validity (Is this a readable, non-duplicate image?)
- `src/labels/` → *semantic* validity (Is the assigned class correct?)
- `src/dcai/` → *distributional* quality (Is the dataset balanced and representative?)

Reporting the incremental effect of each stage separately (baseline → cleaned →
noise_corrected → dcai_improved) produces a much stronger results section than
a single before/after comparison, since it attributes the performance gain to
specific, named interventions.

## 5. Active Learning as a Data Efficiency Argument

The active learning loop (`src/active_learning/`) is evaluated on a different
axis than the other three conditions: instead of asking "does this data
improve final accuracy?", it asks "does this *sampling strategy* reach a given
accuracy with fewer labeled samples?" This produces the labels-used-vs-F1
learning curve, plotted against a random-sampling baseline in
`src/visualization/results_plots.py::plot_active_learning_curve`.

## 6. Known Simplifications (State These Explicitly in the Report)

1. The active learning "oracle" is simulated using existing ground-truth
   labels, not a real human annotator — real deployments would also need to
   model annotation time/cost, which this project approximates as uniform
   per-sample cost.
2. Label noise correction defaults to *removal* rather than *relabeling*
   (see `src/labels/noise_correction.py`), since relabeling risks introducing
   a different kind of error if the model's confidence is miscalibrated.
3. `cleanlab`'s confident learning requires out-of-sample predicted
   probabilities (via cross-validation), which is more expensive than a
   single train/predict pass — this is a deliberate trade-off for
   methodological correctness over speed.
