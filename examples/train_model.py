"""
Example script for training the audio deepfake detector.

Usage:
    python examples/train_model.py --config config.yaml
"""

import argparse
import logging
import numpy as np
from pathlib import Path

from src.models import HybridDeepfakeDetector
from src.training import Trainer
from src.utils import setup_logging, Config


def main(config_path: str = "config.yaml"):
    """Train the deepfake detection model."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("Audio Deepfake Detection - Training")
    logger.info("=" * 50)

    # Load configuration
    config = Config(config_path)
    logger.info(f"Configuration loaded from {config_path}")

    # Create directories
    model_dir = config.get("paths.model_dir", "models")
    log_dir = config.get("paths.log_dir", "logs")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Note: In production, load real data from your dataset
    logger.info("Generating synthetic data for demonstration...")
    # Create dummy data for illustration
    n_train = 100
    n_val = 30
    input_shape = (2, 39, 256)

    X_train = np.random.randn(n_train, *input_shape).astype(np.float32)
    y_train = np.random.randint(0, 2, (n_train, 1)).astype(np.float32)

    X_val = np.random.randn(n_val, *input_shape).astype(np.float32)
    y_val = np.random.randint(0, 2, (n_val, 1)).astype(np.float32)

    logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    # Create model
    logger.info("Creating model...")
    model = HybridDeepfakeDetector(
        input_shape=input_shape,
        num_cnn_filters=config.get("model.num_cnn_filters", 32),
        lstm_units=config.get("model.lstm_units", 128),
        dropout_rate=config.get("model.dropout_rate", 0.3),
        num_attention_heads=config.get("model.num_attention_heads", 8),
    )

    # Create trainer
    trainer = Trainer(model, model_dir=model_dir, log_dir=log_dir)

    # Train model
    logger.info("Starting training...")
    epochs = config.get("training.epochs", 50)
    batch_size = config.get("training.batch_size", 32)
    learning_rate = config.get("training.learning_rate", 0.001)
    patience = config.get("training.patience", 10)

    history = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
    )

    # Save model
    model_path = trainer.save_model()
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = trainer.save_training_history()
    logger.info(f"Training history saved to {history_path}")

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    metrics = trainer.evaluate(X_val, y_val, batch_size=batch_size)

    logger.info("\n" + "=" * 50)
    logger.info("Training complete!")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"History saved: {history_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train audio deepfake detection model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    main(args.config)
