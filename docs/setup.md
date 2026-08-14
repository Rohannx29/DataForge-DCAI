# Setup Instructions

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### GPU Setup (Do This BEFORE installing requirements.txt if you have an NVIDIA GPU)

Check your GPU and max supported CUDA version:
```powershell
nvidia-smi
```

Install the CUDA-enabled PyTorch build FIRST, before requirements.txt (plain
PyPI only hosts CPU-only wheels on Windows):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
If that fails, try `cu121` instead of `cu124`. Your GPU driver is generally
backward-compatible with older CUDA build tags.

Verify:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True` and your GPU name. If it prints `False`, do NOT proceed
to install requirements.txt yet — the CPU wheel will get installed instead
and silently "succeed," but training will run on CPU with no error to warn you.

### Then Install Everything Else

```bash
pip install -r requirements.txt
```

This project was designed to run on a single consumer GPU with 4–6GB VRAM
using ResNet18/EfficientNet-B0. CPU-only execution works for Phase 1 (Casting
dataset, small) but becomes impractical for Phase 2 (MVTec AD) and especially
the active learning loop (repeated retraining across many rounds).

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

**Note:** the Kaggle archive contains two copies of this dataset — a flat,
unsplit `casting_512x512/` directory and a pre-split `casting_data/` directory.
This project's config points at the flat copy only
(`data/raw/casting/casting_512x512/casting_512x512/`) so that train/val/test
splitting stays fully controlled by this repo's own `stratified_split()`
rather than the dataset author's pre-made split. See `docs/dataset_description.md`.

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

# 2. Build the raw 'baseline' condition manifest (no cleaning applied)
python scripts/build_baseline_manifest.py --config configs/dataset_casting.yaml

# 3. Clean data -> produces the 'cleaned' condition manifest
python scripts/run_cleaning.py --config configs/dataset_casting.yaml

# 4. Audit + correct label noise -> produces the 'noise_corrected' condition manifest
python scripts/run_label_audit.py --config configs/dataset_casting.yaml \
    --manifest data/processed/casting/manifest_cleaned.csv

# 5. Train on each condition (repeat --condition for baseline/cleaned/noise_corrected/dcai_improved)
python scripts/train_baseline.py \
    --manifest data/processed/casting/manifest_baseline.csv \
    --condition baseline

# 6. Compare all logged conditions
python scripts/compare_experiments.py --experiment-name defect-dcai
```

## 5. Running Tests

```bash
pytest tests/ -v --cov=src
```