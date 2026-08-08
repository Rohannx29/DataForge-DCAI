"""
Reproducibility utilities.

A fixed seed is applied identically across ALL experimental conditions
(baseline / cleaned / noise_corrected / dcai_improved). This is required
so that performance differences can be attributed to data quality changes
rather than random initialization noise. See docs/architecture.md.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed value. Must be identical across compared experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic cuDNN behavior at a small performance cost.
    # Kept ON for this project since experimental reproducibility outweighs speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
