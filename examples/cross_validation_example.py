"""
Example script for cross-validation.

Usage:
    python examples/cross_validation_example.py
"""

import logging
import numpy as np
from pathlib import Path

from src.models import HybridDeepfakeDetector
from src.training import Trainer
from src.utils import setup_logging, Config


def main(config_path: str = "config.yaml"):
    """Perform cross-validation."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("Audio Deepfake Detection - Cross-Validation")
    logger.info("=" * 50)

    # Load configuration
    config = Config(config_path)
    logger.info(f"Configuration loaded from {config_path}")

    # Create directories
    model_dir = config.get("paths.model_dir", "models")
    log_dir = config.get("paths.log_dir", "logs")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    logger.info("Generating synthetic data for cross-validation...")
    n_samples = 200
    input_shape = (2, 39, 256)

    X = np.random.randn(n_samples, *input_shape).astype(np.float32)
    y = np.random.randint(0, 2, (n_samples, 1)).astype(np.float32)

    logger.info(f"Data shape: {X.shape}, {y.shape}")

    # Create trainer with initial model
    model = HybridDeepfakeDetector(
        input_shape=input_shape,
        num_cnn_filters=config.get("model.num_cnn_filters", 32),
        lstm_units=config.get("model.lstm_units", 128),
        dropout_rate=config.get("model.dropout_rate", 0.3),
    )

    trainer = Trainer(model, model_dir=model_dir, log_dir=log_dir)

    # Run cross-validation
    logger.info("Starting 5-fold cross-validation...")
    n_splits = 5
    epochs = config.get("training.epochs", 20)  # Fewer epochs for CV
    batch_size = config.get("training.batch_size", 32)
    learning_rate = config.get("training.learning_rate", 0.001)

    cv_results = trainer.cross_validate(
        X=X,
        y=y,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Print summary
    avg_metrics = cv_results["average_metrics"]
    logger.info("\nCross-Validation Summary:")
    logger.info(f"  Average Accuracy: {avg_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Average Precision: {avg_metrics.get('precision', 0):.4f}")
    logger.info(f"  Average Recall: {avg_metrics.get('recall', 0):.4f}")
    logger.info(f"  Average F1-Score: {avg_metrics.get('f1_score', 0):.4f}")
    logger.info(f"  Average ROC-AUC: {avg_metrics.get('roc_auc', 0):.4f}")

    logger.info("\n" + "=" * 50)
    logger.info("Cross-validation complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
