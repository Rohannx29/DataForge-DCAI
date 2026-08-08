"""
Logging utility.

Provides a single configured logger used across the pipeline so that
data-processing, training, and evaluation logs are consistent and can be
redirected to a run-specific log file under experiments/logs/.
"""
import logging
from pathlib import Path


def get_logger(name: str, log_dir: str | Path = "experiments/logs", level: str = "INFO") -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.
        log_dir: Directory where log files are written.
        level: Logging level string, e.g. "INFO", "DEBUG".

    Returns:
        Configured logging.Logger instance with console + file handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicate handlers if called multiple times.
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / f"{name.replace('.', '_')}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
