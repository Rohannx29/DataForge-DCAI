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
  but not representative of realistic production-line imbalance.
- **Image properties:** grayscale, 512×512 source resolution, resized to
  224×224 for model input
- **Data quality findings:**
  - 0 corrupt/unreadable files (out of 1,300)
  - 0 duplicate groups within the correctly-scoped `casting_512x512/` copy
    (64 duplicate groups originally found were cross-copy duplicates between
    `casting_512x512/` and `casting_data/`, resolved by scoping `raw_dir`)

## Baseline Model Performance — Ceiling Effect (Casting)

- **Architecture:** ResNet18, pretrained, fine-tuned
- **Test set (196 held-out images), 5 independent runs with varied seeds:**
  test_f1 mean = 0.9974, std = 0.0023, min = 0.9957, max = 1.0000
- **Interpretation:** Near-ceiling performance with negligible variance
  across seeds. Leaves no meaningful headroom to demonstrate data-quality
  intervention gains, confirming Casting's role as Phase 1 (pipeline
  validation) rather than the primary experimental dataset.

## Label Quality Notes (Casting)

### Synthetic Noise Detection Validation

Since Casting's real labels appear near-perfectly clean, the noise-detection
pipeline (`cleanlab`-based confident learning) was validated using
**synthetic label noise** injected at a known rate: 10% of training labels
(90 / 909) deliberately corrupted; 5-fold cross-validation used for
out-of-sample predictions. Result: 89 samples flagged, TP=80, FP=9, FN=10,
**precision=0.899, recall=0.889** — confirming the detection machinery is
correctly wired ahead of MVTec AD.

### Real Label Correction — Effect on Downstream Performance

Applying correction to Casting's REAL labels flagged and removed **1
training sample out of 909 (0.11%)**. 5-run comparison:

| Condition | Test F1 | 95% CI |
|---|---|---|
| baseline | 0.9974 ± 0.0023 | [0.9946, 1.0003] |
| noise_corrected | 0.9923 ± 0.0019 | [0.9899, 0.9947] |

t-test: t=3.803, **p=0.0052** (significant) — but NOT interpretable as a real
capability difference. Only 0.11% of data was removed; the measured Δ
(0.0051 F1) corresponds to roughly one additional misclassified image out of
196 test samples — within the coarsest unit F1 can resolve at this test set
size. Two compounding factors: (1) **metric quantization** — near-ceiling F1
on 196 samples has few discrete values; (2) **training-loader stochasticity**
— `DataLoader(shuffle=True)` produces a different shuffle/batch composition
when dataset length changes (909→908), even under a fixed seed. **Practical
conclusion:** significance tests on near-ceiling metrics with small test
sets can reflect training stochasticity rather than a real data-quality
effect — a methodological caution carried into the MVTec AD analysis below,
where categories are not expected to be at ceiling.

---

## MVTec Anomaly Detection

- **Source:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Task:** Binary classification (good / defective) per object category
- **Categories used:** bottle, metal_nut, screw
- **Methodology adaptation (important):** MVTec AD's official directory
  structure is designed for *unsupervised anomaly detection* — `train/`
  contains ONLY "good" images; all defective images live under
  `test/<defect_type>/`, split across multiple named defect subtypes per
  category (e.g. bottle: `broken_large`, `broken_small`, `contamination`).
  This project POOLS `train/good/` + `test/good/` + all `test/<defect_type>/`
  images per category into a single binary-labeled manifest, then applies
  this repo's own `stratified_split_per_category()` (see
  `src/data/mvtec_preprocessing.py`) to build a genuine, reproducible
  train/val/test classification split — independently within each category,
  so no single category dominates any split. This is a deliberate departure
  from MVTec AD's original anomaly-detection benchmark protocol, made
  explicit here since it affects how results should be compared against
  published MVTec AD anomaly-detection baselines (they are NOT directly
  comparable — different task setup).
- **Total samples:** 1,107 across 3 categories (0 corrupt/unreadable,
  verified against the manifest directly)
- **Class distribution (pooled):**
  | Class | Label | Count |
  |---|---|---|
  | good | 0 | 832 |
  | defective | 1 | 275 |
  - **Imbalance ratio: 3.03** (majority:minority)
- **Class distribution (per category):**
  | Category | Good | Defective | Imbalance Ratio |
  |---|---|---|---|
  | bottle | 229 | 63 | 3.63 |
  | metal_nut | 242 | 93 | 2.60 |
  | screw | 361 | 119 | 3.03 |
- **Defect type diversity (defective samples only):**
  | Category | Defect Types | Samples per Type |
  |---|---|---|
  | bottle | broken_large, broken_small, contamination | 20, 22, 21 |
  | metal_nut | bent, color, flip, scratch | 25, 22, 23, 23 |
  | screw | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top | 24, 24, 25, 23, 23 |
- **Split sizes (per category, ~70/15/15):**
  | Category | Train | Val | Test |
  |---|---|---|---|
  | bottle | 204 | 44 | 44 |
  | metal_nut | 234 | 50 | 51 |
  | screw | 336 | 72 | 72 |
- **Contrast with Casting (critical for report narrative):** Casting had
  defective as the MAJORITY class at a mild 1.50:1 ratio; MVTec AD has
  defective as the genuine MINORITY class at 3.03:1 pooled (up to 3.63:1 for
  bottle) — far closer to realistic manufacturing defect rates, and the
  dataset this project's class-balance and targeted-augmentation modules
  (`src/dcai/class_balance.py`, `src/dcai/augmentation.py`) are designed to
  address. This is the direct motivation for scoping MVTec AD as the primary
  experimental comparison (Phase 2), with Casting serving only as pipeline
  validation (Phase 1).
- **Image properties:** varies by category (bottle/metal_nut RGB, screw
  grayscale in source), all converted to RGB and resized to 224×224 for
  model input (see `src/data/dataset.py::DefectDataset`)

## Label Quality Notes (MVTec AD)

_To be populated after running the noise-detection pipeline on this dataset
(Phase 2) — expect this to be more informative than Casting's near-zero
finding, since MVTec AD's real defect labels were assigned by domain experts
following a documented protocol rather than the Casting dataset's uncertain
labeling provenance._