#!/usr/bin/env python
"""
Entry point: dataset acquisition.

Usage:
    python scripts/download_data.py --dataset casting
    python scripts/download_data.py --dataset mvtec_ad
"""
import argparse

from src.data.acquisition import download_casting_dataset, download_mvtec_ad
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a raw dataset")
    parser.add_argument("--dataset", choices=["casting", "mvtec_ad"], required=True)
    args = parser.parse_args()

    if args.dataset == "casting":
        download_casting_dataset()
    elif args.dataset == "mvtec_ad":
        download_mvtec_ad()

    logger.info("Download complete for dataset: %s", args.dataset)


if __name__ == "__main__":
    main()
