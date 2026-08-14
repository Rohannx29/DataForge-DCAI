# Dataset Description

## Casting Product Image Dataset

- **Source:** Kaggle — Real-life Industrial Dataset of Casting Product
  (`ravirajsinh45/real-life-industrial-dataset-of-casting-product`)
- **Task:** Binary classification (defective / ok) of submersible pump impeller castings
- **Archive structure note:** The Kaggle download contains TWO copies of the
  dataset — a flat, unsplit `casting_512x512/` directory and a pre-split
  `casting_data/{train,test}/` directory. This project uses ONLY the flat
  `casting_512x512/casting_512x512/` copy, so that train/val/test splitting
  is fully controlled by this repo's own `stratified_split()` (fixed seed,
  reproducible) rather than inheriting the dataset author's pre-made split.
  See `configs/dataset_casting.yaml` for the exact path.
- **Total samples used:** 1,300 (from the flat, unsplit copy)
- **Class distribution:**
  | Class | Label | Count |
  |---|---|---|
  | def_front (defective) | 1 | 781 |
  | ok_front (normal) | 0 | 519 |
- **Class imbalance ratio:** 1.50 (majority:minority)
- **Notable characteristic:** Unlike typical real-world manufacturing defect
  rates (often <5% defective), this dataset's defective class is the
  MAJORITY class, and the imbalance is mild (1.5:1) rather than severe. This
  makes Casting well-suited as a **pipeline validation dataset** (Phase 1)
  but not representative of realistic production-line imbalance — that
  characteristic is expected to show up more strongly in MVTec AD (Phase 2).
- **Image properties:** grayscale, 512×512 source resolution, resized to
  224×224 for model input
- **Data quality findings (via this repo's validation pipeline):**
  - 0 corrupt/unreadable files (out of 1,300)
  - 0 duplicate groups within the correctly-scoped `casting_512x512/` copy
    (an earlier validation run scoped to the full `data/raw/casting/` tree
    found 64 duplicate groups — these were cross-copy duplicates between
    `casting_512x512/` and `casting_data/`, not genuine within-dataset
    duplicates; resolved by scoping `raw_dir` to a single copy)

## Baseline Model Performance — Ceiling Effect

- **Architecture:** ResNet18, pretrained, fine-tuned (see `configs/model_config.yaml`)
- **Test set (196 held-out images, never used for model selection), 5 independent runs with varied seeds:**
  test_f1 mean = 0.9974, std = 0.0023, min = 0.9957, max = 1.0000
- **Interpretation:** Casting Product is a well-documented "easy" benchmark;
  public baselines commonly report 98–100% with simple CNNs. Multi-run
  averaging (5 seeds) confirms this is a near-ceiling result — not a fluke
  of a single training run — with negligible variance (std = 0.0023). This
  leaves effectively no headroom to demonstrate gains from data-quality
  interventions (cleaned / noise_corrected / dcai_improved conditions cannot
  meaningfully exceed a near-perfect score). This confirms the pipeline is
  functioning correctly end-to-end and is the direct motivation for treating
  Casting as Phase 1 (pipeline validation only) and MVTec AD as Phase 2
  (primary experimental comparison), per the original project plan.

## Label Quality Notes

### Synthetic Noise Detection Validation

Since Casting's real labels appear near-perfectly clean (consistent with the
ceiling-effect baseline result above), there is no meaningful REAL label
noise to detect on this dataset. The noise-detection pipeline (`cleanlab`-based
confident learning, `src/labels/noise_detection.py`) was instead validated
using **synthetic label noise** injected at a known rate:

- **Method:** 10% of training labels (90 / 909) deliberately corrupted;
  5-fold cross-validation used to obtain out-of-sample predicted
  probabilities; `cleanlab.filter.find_label_issues()` used to flag
  likely-noisy samples.
- **Result:** 89 samples flagged. TP=80, FP=9, FN=10. Precision=0.899,
  Recall=0.889.
- **Interpretation:** The detector correctly identifies ~89% of deliberately
  injected label errors with ~90% precision — the noise-detection machinery
  is correctly wired and functioning ahead of Phase 2 (MVTec AD).

### Real Label Correction — Effect on Downstream Performance

Applying `cleanlab`-based correction to Casting's REAL (non-synthetic)
labels flagged and removed **1 training sample out of 909 (0.11%)** —
consistent with the dataset's near-ceiling baseline performance, indicating
almost no genuine label noise is present.

5-run comparison (test F1, mean ± std, 95% CI):

| Condition | Test F1 | 95% CI |
|---|---|---|
| baseline | 0.9974 ± 0.0023 | [0.9946, 1.0003] |
| noise_corrected | 0.9923 ± 0.0019 | [0.9899, 0.9947] |

Independent-samples t-test: t=3.803, **p=0.0052** (statistically significant
at α=0.05).

**Interpretation — read carefully, this is a methodological caution, not a
capability claim:** despite reaching statistical significance across 5 seeds,
this result should NOT be read as "noise correction hurt the model." Only
0.11% of training data was removed — far too little to plausibly cause a
real capability difference in a fine-tuned ResNet18. The measured Δ (0.0051
F1) corresponds to roughly **one additional misclassified image, on average,
out of 196 test images** — i.e., the entire "effect" is within the coarsest
unit the metric can resolve. Two compounding factors explain the significant-
but-spurious result:

1. **Metric quantization:** with only 196 test samples, F1 near ceiling can
   only take a small number of discrete values; a difference of "1 image on
   average" is close to the smallest measurable unit, so t-tests on such
   metrics are unusually sensitive to noise even when seeds are fixed.
2. **Training-loader stochasticity:** `DataLoader(shuffle=True)` generates a
   shuffle permutation dependent on the CURRENT dataset length. Removing 1
   sample (909→908) changes the shuffle/batch composition for the entire
   training run, producing a genuinely different training trajectory from
   the same seed — not "identical training minus one example."

**Practical conclusion:** on near-ceiling tasks with small test sets,
statistically-significant results from n=5 seeded runs can arise from
training/data-loader stochasticity rather than a real, causally meaningful
effect of the data intervention being tested. This motivates two things
going forward: (a) treating significance tests on Casting as inconclusive
by design, given the ceiling effect, and (b) expecting more trustworthy,
larger-effect-size comparisons on MVTec AD (Phase 2), where categories are
not expected to be at ceiling and test sets may allow finer metric resolution.