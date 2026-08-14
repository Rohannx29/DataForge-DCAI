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
[unchanged from before — precision=0.899, recall=0.889]

### Real Label Correction — Effect on Downstream Performance

Applying `cleanlab`-based noise correction to Casting's real (non-synthetic)
labels removed <N> training samples (<X.XX>% of the 909-sample training set)
flagged as likely mislabeled. Downstream 5-run comparison against the
uncorrected baseline:

| Condition | Test F1 (mean ± std) | 95% CI |
|---|---|---|
| baseline | 0.9974 ± 0.0023 | [0.9946, 1.0003] |
| cleaned | 0.9974 ± 0.0023 | [0.9946, 1.0003] |
| noise_corrected | 0.9923 ± 0.0019 | [0.9899, 0.9947] |

An independent-samples t-test (baseline vs. noise_corrected, n=5 each)
confirms this difference is statistically significant: t=3.803, p=0.0052.

**Interpretation:** Noise correction produced a statistically significant
*decrease* in test performance on this dataset. Since baseline already
achieves near-ceiling performance (0.997 test F1), Casting's real labels
contain very little genuine noise for `cleanlab` to correctly identify.
Under these conditions, the detector's false positives (correctly-labeled
samples incorrectly flagged and removed — measured at 9/89, ~10% FP rate,
in the synthetic noise validation above) dominate its impact: removing valid
training data shrinks an already-small training set (909 samples) with no
compensating noise-reduction benefit. This demonstrates that label-noise
correction is not universally beneficial — its value is contingent on the
actual noise rate present in the source data, motivating close attention to
this dynamic when the same pipeline is applied to MVTec AD in Phase 2, where
the true noise rate is unknown and may differ substantially.

**baseline vs. cleaned are bit-identical** (mean, std, CI all match exactly)
— expected, not a bug: `manifest_cleaned.csv` is byte-identical to
`manifest_baseline.csv` for this dataset (0 corrupt files, 0 duplicates
found within the correctly-scoped raw directory — see Data quality findings
above), and both conditions share the same seed sequence with deterministic
cuDNN settings enabled (`src/utils/seed.py`), producing bit-reproducible
results. This will not hold for MVTec AD, where cleaning is expected to
meaningfully alter the dataset.