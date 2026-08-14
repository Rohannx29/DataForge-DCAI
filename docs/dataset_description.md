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
- **Test set (196 held-out images, never used for model selection):**
  accuracy = 1.0, precision = 1.0, recall = 1.0, F1 = 1.0, AUROC = 1.0
- **Interpretation:** Casting Product is a well-documented "easy" benchmark;
  public baselines commonly report 98–100% with simple CNNs. A perfect
  baseline score means **no headroom exists** on this dataset to demonstrate
  gains from data-quality interventions (cleaned / noise_corrected /
  dcai_improved conditions cannot exceed a perfect score). This confirms the
  pipeline is functioning correctly end-to-end but is the direct motivation
  for treating Casting as Phase 1 (pipeline validation only) and MVTec AD as
  Phase 2 (primary experimental comparison), per the original project plan.

## Label Quality Notes

Since Casting's real labels appear near-perfectly clean (consistent with the
ceiling-effect baseline result above), there is no meaningful REAL label
noise to detect on this dataset. Instead, the noise-detection pipeline
(`cleanlab`-based confident learning, `src/labels/noise_detection.py`) was
validated using **synthetic label noise** injected at a known rate:

- **Method:** 10% of training labels (90 / 909) deliberately corrupted via
  `inject_synthetic_label_noise()`; 5-fold cross-validation used to obtain
  out-of-sample predicted probabilities (`get_out_of_sample_predictions()`);
  `cleanlab.filter.find_label_issues()` used to flag likely-noisy samples.
- **Result:** 89 samples flagged as likely mislabeled.
  - True Positives: 80 (correctly identified injected noise)
  - False Positives: 9 (real labels incorrectly flagged)
  - False Negatives: 10 (injected noise missed)
  - **Precision: 0.899, Recall: 0.889**
- **Interpretation:** The noise-detection pipeline correctly identifies
  ~89% of deliberately injected label errors with ~90% precision, validating
  that `cleanlab`-based confident learning is correctly wired into this
  project's k-fold cross-validation pipeline ahead of applying it to MVTec
  AD's real (uninjected) label noise in Phase 2, where actual noisy labels
  are expected.
- **Known implementation note:** `cleanlab.filter.find_label_issues()` is
  forced to `n_jobs=1` (single-process) rather than its multiprocessing
  default, due to a Windows + Python 3.12 + `torch._dynamo` compatibility
  bug where subprocess re-imports of `torch` trigger a `MemoryError` via a
  runaway `inspect.signature()` recursion. This is a platform-specific
  workaround, not a change to the underlying detection methodology.