# Setup Instructions

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+ and, ideally, a CUDA-capable GPU (project was designed
to run on a single consumer GPU with 4–6GB VRAM using ResNet18/EfficientNet-B0).
CPU-only execution is supported but significantly slower for the active
learning loop (repeated retraining).

## 2. Dataset Acquisition

### Casting Product Image Dataset (Phase 1)

Fully automated via `kagglehub`. Requires a Kaggle account and API token:
1. Create an API token at https://www.kaggle.com/settings → "Create New Token"
   (downloads a `kaggle.json` file)
2. Place it at `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
3. Run:
```bash
   python scripts/download_data.py --dataset casting
```

This downloads (or reuses a cached copy from `~/.cache/kagglehub`), copies the
dataset into `data/raw/casting/`, and verifies the file count looks complete —
it will raise a clear error instead of silently proceeding if the download is
partial or corrupted.

### MVTec Anomaly Detection (Phase 2)

MVTec AD requires manually accepting the license terms on the official page
before download (cannot be automated):
1. Visit https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Accept the license and download the categories listed in
   `configs/dataset_mvtec.yaml` (`bottle`, `metal_nut`, `screw` by default)
3. Extract into `data/raw/mvtec_ad/<category>/`

## 3. Experiment Tracking

This project uses MLflow by default (local, no account needed):

```bash
mlflow ui --backend-store-uri experiments/mlruns
```

Then open http://localhost:5000 to view logged runs. To use Weights & Biases
instead, change `experiment_tracking.backend` in `configs/base_config.yaml`
and set `WANDB_API_KEY` in a `.env` file (see `.env.local` — not committed).

## 4. Running the Pipeline

```bash
# 1. Validate raw data
python scripts/run_validation.py --config configs/dataset_casting.yaml

# 2. Clean data -> produces the 'cleaned' condition manifest
python scripts/run_cleaning.py --config configs/dataset_casting.yaml

# 3. Audit + correct label noise -> produces the 'noise_corrected' condition manifest
python scripts/run_label_audit.py --config configs/dataset_casting.yaml \
    --manifest data/processed/casting/manifest_cleaned.csv

# 4. Train on each condition (repeat --condition for baseline/cleaned/noise_corrected/dcai_improved)
python scripts/train_baseline.py \
    --manifest data/processed/casting/manifest_cleaned.csv \
    --condition cleaned

# 5. Compare all logged conditions
python scripts/compare_experiments.py --experiment-name defect-dcai
```

## 5. Running Tests

```bash
pytest tests/ -v --cov=src
```