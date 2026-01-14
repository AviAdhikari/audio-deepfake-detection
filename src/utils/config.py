"""Configuration management."""

import yaml
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for the deepfake detector."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML or JSON config file
        """
        self.config = self._load_default_config()

        if config_path:
            self.load_config(config_path)

    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        return {
            "audio": {
                "target_sr": 16000,
                "n_mels": 128,
                "n_fft": 2048,
                "hop_length": 512,
                "n_mfcc": 13,
                "target_duration": None,
                "target_length": 256,
            },
            "model": {
                "num_cnn_filters": 32,
                "lstm_units": 128,
                "dropout_rate": 0.3,
                "num_attention_heads": 8,
                "input_shape": (2, 39, 256),
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 10,
                "validation_split": 0.2,
                "test_split": 0.1,
            },
            "inference": {
                "threshold": 0.5,
                "batch_size": 32,
            },
            "paths": {
                "data_dir": "data",
                "model_dir": "models",
                "log_dir": "logs",
                "output_dir": "output",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        }

    def load_config(self, config_path: str):
        """
        Load configuration from file.

        Args:
            config_path: Path to YAML or JSON config file
        """
        try:
            with open(config_path, "r") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    custom_config = yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    custom_config = json.load(f)
                else:
                    raise ValueError("Config file must be YAML or JSON")

            # Merge with default config
            self._deep_merge(self.config, custom_config)
            logger.info(f"Configuration loaded from {config_path}")

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            raise

    def save_config(self, output_path: str):
        """
        Save current configuration to file.

        Args:
            output_path: Path to save config
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if str(output_path).endswith(".yaml") or str(output_path).endswith(".yml"):
                with open(output_path, "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                with open(output_path, "w") as f:
                    json.dump(self.config, f, indent=2)

            logger.info(f"Configuration saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

    def get(self, key: str, default=None):
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "audio.target_sr")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value):
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "audio.target_sr")
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")

    def to_dict(self) -> Dict:
        """Get configuration as dictionary."""
        return self.config.copy()

    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2)

    @staticmethod
    def _deep_merge(base: Dict, update: Dict):
        """Recursively merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value
