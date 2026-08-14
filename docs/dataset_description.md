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
  characteristic is expected to show up more strongly in MVTec AD (Phase 2),
  where each category is evaluated independently and defect rates vary.
- **Image properties:** grayscale, 512×512 source resolution, resized to
  224×224 for model input
- **Data quality findings (via this repo's validation pipeline):**
  - 0 corrupt/unreadable files (out of 1,300)
  - 0 duplicate groups within the correctly-scoped `casting_512x512/` copy
    (an earlier validation run scoped to the full `data/raw/casting/` tree
    found 64 duplicate groups — these were cross-copy duplicates between
    `casting_512x512/` and `casting_data/`, not genuine within-dataset
    duplicates; resolved by scoping `raw_dir` to a single copy)

## MVTec Anomaly Detection

- **Source:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Task:** Binary classification (good / defective) per object category; pixel-level
  defect masks available for future segmentation work
- **Categories used:** bottle, metal_nut, screw (see `configs/dataset_mvtec.yaml`)
- **Total samples per category:** TODO (populate after Phase 2 download)
- **Class distribution:** TODO — MVTec AD training sets contain ONLY "good"
  samples by design (anomaly detection setup); defective samples appear only
  in the test split. **This has methodology implications** — document how
  this project adapts MVTec AD's original anomaly-detection framing into the
  supervised classification setup used here (likely requires re-splitting
  test-set defective images into train/val/test rather than using the
  original AD splits directly).
- **Known limitations:** TODO

## Label Quality Notes

_After running `scripts/run_label_audit.py`, document here:_
- Number/percentage of samples flagged by `cleanlab`
- Manual review findings for a sample of flagged images (was the flag correct?)
- Any inter-annotator agreement statistics if a re-labeling audit was performed