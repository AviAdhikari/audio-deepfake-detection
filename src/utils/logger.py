"""Logging configuration."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_format: str = None,
    log_file: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        log_file: Whether to save logs to file

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"deepfake_detector_{timestamp}.log"
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
