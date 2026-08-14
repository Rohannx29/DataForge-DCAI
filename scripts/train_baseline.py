#!/usr/bin/env python
"""
Entry point: train a model for a single experimental condition, repeated
n_runs times with varied random seeds to produce a distribution of results
(mean, std, 95% CI) rather than a single point estimate.

This script is intentionally the SAME regardless of which condition is being
run — the --manifest and --condition arguments determine what data is used
and how each run is tagged; model architecture/hyperparameters never change
across conditions (see docs/architecture.md).

Usage:
    python scripts/train_baseline.py \
        --manifest data/processed/casting/manifest_baseline.csv \
        --condition baseline \
        --model-config configs/model_config.yaml \
        --base-config configs/base_config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import DefectDataset
from src.experiment.tracker import log_metrics, log_params, track_run
from src.models.architectures import build_model
from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.utils.config import load_config, merge_configs
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def run_single_training(
    run_idx: int,
    run_seed: int,
    manifest_path: str,
    condition: str,
    config: dict,
    checkpoint_dir: Path,
    transform,
) -> dict[str, float]:
    """Train and evaluate one model instance (one of n_runs repeats).

    Args:
        run_idx: 0-indexed run number, used in checkpoint filename and MLflow tags.
        run_seed: Seed for THIS run's model initialization (base_seed + run_idx).
        manifest_path: Path to the condition's manifest CSV.
        condition: One of baseline/cleaned/noise_corrected/dcai_improved.
        config: Merged base_config + model_config.
        checkpoint_dir: Directory to save this run's model weights.
        transform: torchvision transform pipeline (shared across runs).

    Returns:
        Dict with both val_* (selection) and test_* (generalization) metrics for this run.
    """
    set_seed(run_seed)

    train_dataset = DefectDataset(manifest_path, split="train", transform=transform)
    val_dataset = DefectDataset(manifest_path, split="val", transform=transform)
    test_dataset = DefectDataset(manifest_path, split="test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
        dropout=config["model"]["dropout"],
    )

    with track_run(config["experiment_tracking"]["experiment_name"], condition=condition, run_name=f"{condition}_run{run_idx}"):
        log_params({
            "condition": condition,
            "run_index": run_idx,
            "run_seed": run_seed,
            "architecture": config["model"]["architecture"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
        })

        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["training"]["epochs"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            device=config["project"]["device"],
            early_stopping_patience=config["training"]["early_stopping_patience"],
        )

        # Val metrics: used for early stopping / model selection during training.
        # NOT a fair generalization estimate — logged for transparency only.
        val_metrics = evaluate_model(model, val_loader, device=config["project"]["device"])

        # Test metrics: the TRUE generalization estimate. Test set is never
        # touched during training or model selection for this run.
        test_metrics = evaluate_model(model, test_loader, device=config["project"]["device"])

        all_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        all_metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        log_metrics(all_metrics)

        logger.info(
            "[%s run %d/%d, seed=%d] val_f1=%.4f | test_f1=%.4f | test_auroc=%.4f",
            condition, run_idx + 1, config["evaluation"]["n_runs"], run_seed,
            val_metrics["f1"], test_metrics["f1"], test_metrics["auroc"],
        )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{condition}_run{run_idx}_model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model for one experimental condition, n_runs times")
    parser.add_argument("--manifest", required=True, help="Path to the manifest CSV for this condition")
    parser.add_argument("--condition", required=True, choices=["baseline", "cleaned", "noise_corrected", "dcai_improved"])
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--base-config", default="configs/base_config.yaml")
    parser.add_argument("--checkpoint-dir", default="experiments/checkpoints", help="Where to save trained model weights")
    parser.add_argument("--n-runs", type=int, default=None, help="Override evaluation.n_runs from config")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    model_cfg = load_config(args.model_config)
    config = merge_configs(base_cfg, model_cfg)

    n_runs = args.n_runs if args.n_runs is not None else config["evaluation"]["n_runs"]
    base_seed = config["project"]["seed"]
    checkpoint_dir = Path(args.checkpoint_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info("Starting %d training runs for condition '%s' (base_seed=%d)", n_runs, args.condition, base_seed)

    all_run_metrics = []
    for run_idx in range(n_runs):
        run_seed = base_seed + run_idx
        metrics = run_single_training(
            run_idx=run_idx,
            run_seed=run_seed,
            manifest_path=args.manifest,
            condition=args.condition,
            config=config,
            checkpoint_dir=checkpoint_dir,
            transform=transform,
        )
        all_run_metrics.append(metrics)

    test_f1_values = [m["test_f1"] for m in all_run_metrics]
    logger.info(
        "[%s] Completed %d runs | test_f1 mean=%.4f std=%.4f min=%.4f max=%.4f",
        args.condition, n_runs, np.mean(test_f1_values), np.std(test_f1_values, ddof=1) if n_runs > 1 else 0.0,
        np.min(test_f1_values), np.max(test_f1_values),
    )
    logger.info(
        "Next: run `python scripts/compare_experiments.py` once all conditions have %d+ runs logged.", n_runs
    )


if __name__ == "__main__":
    main()