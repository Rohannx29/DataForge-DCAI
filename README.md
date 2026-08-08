# Data-Centric Active Learning for Manufacturing Defect Detection

A reproducible research pipeline that investigates how **data quality interventions**
(label-noise correction, targeted augmentation, active-learning-guided labeling) affect
defect classification performance, while holding the model architecture constant.

## Core Research Question

> Given a fixed CNN classifier, does improving the *data* (labels, sampling strategy,
> augmentation) produce larger performance gains than improving the *model*?

This is tested by running the identical architecture across four experimental conditions:

1. **Baseline** — raw dataset, random label sampling, no cleaning
2. **Cleaned** — corrupt/duplicate/invalid samples removed
3. **Noise-Corrected** — mislabeled samples detected (via `cleanlab`) and corrected/removed
4. **DCAI-Improved** — active-learning-guided label acquisition + targeted augmentation
   for rare defect classes

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| Casting Product Image Dataset | Phase 1 — pipeline validation | Binary (defective/ok), simple structure |
| MVTec AD | Phase 2 — main experiments | Multi-category, pixel-level defect masks, industry benchmark |

Datasets are **not** committed to this repository. See `docs/setup.md` for download
instructions; `scripts/download_data.py` automates acquisition where licensing allows.

## Repository Structure

```
defect-dcai/
├── configs/            # YAML configs: dataset, model, training, active learning
├── data/
│   ├── raw/            # Immutable original data — never modified
│   ├── interim/        # Intermediate, partially processed data
│   ├── processed/      # Final, model-ready data
│   └── external/       # Third-party reference data (if any)
├── src/
│   ├── data/           # Acquisition, dataset classes, validation, cleaning, preprocessing
│   ├── labels/         # Label quality audit, noise detection & correction
│   ├── models/         # Architecture, training loop, evaluation (kept stable/controlled)
│   ├── active_learning/# Sampling strategies, the AL query loop
│   ├── dcai/           # Targeted augmentation, outlier detection, class balancing
│   ├── experiment/     # Experiment tracking wrapper, cross-condition comparison
│   ├── visualization/  # EDA plots, results plots, Grad-CAM
│   └── utils/          # Config loading, seeding, logging
├── scripts/            # Thin CLI entry points calling into src/
├── tests/              # Unit tests for data/label logic
├── notebooks/          # Exploratory analysis only — never imported by src/
├── experiments/        # Run outputs, metrics, artifacts (git-ignored)
├── docs/               # Architecture, setup, dataset description
└── reports/            # Literature survey, final report, figures
```

## Design Principle: Controlled Experimentation

The model architecture (`src/models/architectures.py`), hyperparameters, random seed,
and training loop are **held fixed** across all four experimental conditions. Only the
`src/data/`, `src/labels/`, and `src/dcai/` layers change between conditions. This is
what allows performance differences to be attributed to data quality rather than
confounded by model changes — see `docs/architecture.md` for the full rationale.

## Setup

See [`docs/setup.md`](docs/setup.md) for environment setup and dataset acquisition.

Quick start:
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py --dataset casting
python scripts/run_validation.py --config configs/dataset_casting.yaml
python scripts/train_baseline.py --config configs/model_config.yaml
```

## Status

🚧 Repository skeleton established. Baseline pipeline implementation in progress.
See `reports/final_report.md` for the current project structure and progress log.

## License

Academic project. Third-party datasets retain their original licenses (see `docs/setup.md`).
