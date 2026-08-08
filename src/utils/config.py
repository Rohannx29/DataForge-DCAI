"""
Configuration loading utilities.

Centralizes YAML config parsing so every script/module reads configuration
the same way, and so config values can be overridden via CLI without editing
YAML files directly (useful for running sweeps across experimental conditions).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a single YAML config file into a dictionary.

    Args:
        config_path: Path to a .yaml config file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an override config into a base config.

    Used to combine base_config.yaml with a dataset- or model-specific config,
    e.g. merge_configs(load_config("base_config.yaml"), load_config("model_config.yaml")).

    Args:
        base: Base configuration dictionary.
        override: Configuration whose values take precedence on conflicts.

    Returns:
        Merged configuration dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
