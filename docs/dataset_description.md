# Dataset Description

_Populate this file with actual statistics after running `scripts/run_eda.py`
on each dataset — this becomes the "Dataset Description" section of the final
report. Placeholder structure below._

## Casting Product Image Dataset

- **Source:** [Kaggle — Real-life Industrial Dataset of Casting Product](TODO: add URL)
- **Task:** Binary classification (defective / ok) of submersible pump impeller castings
- **Total samples:** TODO (run `scripts/run_eda.py`)
- **Class distribution:** TODO
- **Image properties:** grayscale, TODO resolution
- **Known limitations:** TODO (document after EDA — e.g. lighting consistency, camera angle variation)

## MVTec Anomaly Detection

- **Source:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Task:** Binary classification (good / defective) per object category; pixel-level
  defect masks available for future segmentation work
- **Categories used:** bottle, metal_nut, screw (see `configs/dataset_mvtec.yaml`)
- **Total samples per category:** TODO
- **Class distribution:** TODO — MVTec AD training sets contain ONLY "good" samples
  by design (anomaly detection setup); defective samples appear only in the test
  split. **This has methodology implications** — document how this project adapts
  MVTec AD's original anomaly-detection framing into the supervised classification
  setup used here (likely requires re-splitting test-set defective images into
  train/val/test rather than using the original AD splits directly).
- **Known limitations:** TODO

## Label Quality Notes

_After running `scripts/run_label_audit.py`, document here:_
- Number/percentage of samples flagged by `cleanlab`
- Manual review findings for a sample of flagged images (was the flag correct?)
- Any inter-annotator agreement statistics if a re-labeling audit was performed
