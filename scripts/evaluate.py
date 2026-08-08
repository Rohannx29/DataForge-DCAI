#!/usr/bin/env python
"""
Entry point: evaluate a trained model on the held-out test set.

Usage:
    python scripts/evaluate.py --manifest data/processed/casting/manifest_baseline.csv --checkpoint path/to/model.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import DefectDataset
from src.models.architectures import build_model
from src.models.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved model state_dict (.pt)")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    args = parser.parse_args()

    config = load_config(args.model_config)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = DefectDataset(args.manifest, split="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    metrics = evaluate_model(model, test_loader, device=config["project"]["device"])
    logger.info("Test set metrics: %s", metrics)


if __name__ == "__main__":
    main()