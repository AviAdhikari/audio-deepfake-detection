"""Example: Training with Transformer models."""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.audio_processor import AudioProcessor
from src.models.transformer_model import TransformerDeepfakeDetector, HybridTransformerCNNDetector
from src.training.trainer import Trainer
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_synthetic_data(n_samples: int = 100, input_shape: tuple = (2, 39, 256)):
    """Generate synthetic audio features for demonstration."""
    logger.info(f"Generating {n_samples} synthetic samples...")

    X = np.random.randn(n_samples, *input_shape).astype(np.float32)
    # Make data more realistic: normalize
    X = (X - X.mean(axis=(1, 2, 3), keepdims=True)) / (
        X.std(axis=(1, 2, 3), keepdims=True) + 1e-7
    )

    # Create labels: roughly 50% deepfake, 50% genuine
    y = np.random.randint(0, 2, n_samples).astype(np.float32)

    logger.info(f"Generated data shape: {X.shape}, labels shape: {y.shape}")
    return X, y


def example_transformer_training():
    """Example: Training TransformerDeepfakeDetector."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Transformer Model Training")
    logger.info("=" * 60)

    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=200)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    logger.info("Creating TransformerDeepfakeDetector...")
    model = TransformerDeepfakeDetector(
        input_shape=(2, 39, 256),
        num_transformer_blocks=2,  # Fewer blocks for faster training
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
    )

    logger.info(f"Model created with {model.count_params()} parameters")

    # Create trainer
    config = ConfigManager()
    config.set("model.type", "transformer")
    config.set("training.epochs", 5)
    config.set("training.batch_size", 16)

    trainer = Trainer(config)

    # Train
    logger.info("Starting training...")
    history, best_model = trainer.train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    logger.info("Training completed!")
    logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")


def example_hybrid_transformer_training():
    """Example: Training HybridTransformerCNNDetector."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Hybrid Transformer-CNN Training")
    logger.info("=" * 60)

    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=200)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    logger.info("Creating HybridTransformerCNNDetector...")
    model = HybridTransformerCNNDetector(
        input_shape=(2, 39, 256),
        num_transformer_blocks=2,
        embed_dim=128,
        num_heads=8,
        ff_dim=256,
    )

    logger.info(f"Model created with {model.count_params()} parameters")

    # Create trainer
    config = ConfigManager()
    config.set("model.type", "hybrid_transformer_cnn")
    config.set("training.epochs", 5)
    config.set("training.batch_size", 16)

    trainer = Trainer(config)

    # Train
    logger.info("Starting training...")
    history, best_model = trainer.train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    logger.info("Training completed!")
    logger.info(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")


def example_model_comparison():
    """Example: Compare Transformer vs original Hybrid model."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Model Architecture Comparison")
    logger.info("=" * 60)

    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=50)

    from src.models.hybrid_model import HybridDeepfakeDetector

    # Create models
    models = {
        "TransformerDeepfakeDetector": TransformerDeepfakeDetector(
            input_shape=(2, 39, 256), num_transformer_blocks=2
        ),
        "HybridTransformerCNN": HybridTransformerCNNDetector(
            input_shape=(2, 39, 256), num_transformer_blocks=2
        ),
        "OriginalHybridCNNLSTM": HybridDeepfakeDetector(input_shape=(2, 39, 256)),
    }

    logger.info("Model parameter comparison:")
    logger.info("-" * 60)

    for name, model in models.items():
        params = model.count_params()
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Quick inference test
        predictions = model.predict(X[:5], verbose=0)

        logger.info(f"{name:30} | Parameters: {params:,}")

    logger.info("-" * 60)


def example_inference_comparison():
    """Example: Compare inference on different model types."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Inference Speed Comparison")
    logger.info("=" * 60)

    import time

    X, _ = generate_synthetic_data(n_samples=32)

    models = {
        "Transformer": TransformerDeepfakeDetector(input_shape=(2, 39, 256), num_transformer_blocks=1),
        "HybridTransformerCNN": HybridTransformerCNNDetector(input_shape=(2, 39, 256), num_transformer_blocks=1),
    }

    logger.info("Inference speed comparison (32 samples):")
    logger.info("-" * 60)

    for name, model in models.items():
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Warmup
        _ = model.predict(X[:1], verbose=0)

        # Benchmark
        start = time.time()
        predictions = model.predict(X, verbose=0)
        elapsed = time.time() - start

        logger.info(f"{name:30} | Time: {elapsed*1000:.2f}ms | Throughput: {len(X)/elapsed:.0f} samples/sec")

    logger.info("-" * 60)


if __name__ == "__main__":
    # Run examples
    example_transformer_training()
    print("\n")

    example_hybrid_transformer_training()
    print("\n")

    example_model_comparison()
    print("\n")

    example_inference_comparison()

    logger.info("All transformer examples completed!")
