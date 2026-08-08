#!/usr/bin/env python
"""
Entry point: train a model for a single experimental condition.

This script is intentionally the SAME regardless of which condition is being
run — the --manifest and --condition arguments determine what data is used
and how the run is tagged; model/training code never changes.

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model for one experimental condition")
    parser.add_argument("--manifest", required=True, help="Path to the manifest CSV for this condition")
    parser.add_argument("--condition", required=True, choices=["baseline", "cleaned", "noise_corrected", "dcai_improved"])
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--base-config", default="configs/base_config.yaml")
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    model_cfg = load_config(args.model_config)
    config = merge_configs(base_cfg, model_cfg)

    set_seed(config["project"]["seed"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DefectDataset(args.manifest, split="train", transform=transform)
    val_dataset = DefectDataset(args.manifest, split="val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
        dropout=config["model"]["dropout"],
    )

    with track_run(config["experiment_tracking"]["experiment_name"], condition=args.condition, run_name=f"{args.condition}_run"):
        log_params({
            "condition": args.condition,
            "architecture": config["model"]["architecture"],
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "seed": config["project"]["seed"],
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

        final_metrics = evaluate_model(model, val_loader, device=config["project"]["device"])
        log_metrics(final_metrics)
        logger.info("Final validation metrics [%s]: %s", args.condition, final_metrics)


if __name__ == "__main__":
    main()