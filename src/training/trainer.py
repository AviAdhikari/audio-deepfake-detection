"""Model training pipeline."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import json
from datetime import datetime

from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training and evaluation."""

    def __init__(
        self,
        model,
        model_dir: str = "models",
        log_dir: str = "logs",
    ):
        """
        Initialize trainer.

        Args:
            model: Keras model to train
            model_dir: Directory to save model checkpoints
            log_dir: Directory to save logs and metrics
        """
        self.model = model
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)

        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = MetricsCalculator()
        self.training_history = {}

    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = "binary_crossentropy",
        metrics: list = None,
    ):
        """
        Compile model with optimizer and loss.

        Args:
            learning_rate: Learning rate for Adam optimizer
            loss: Loss function
            metrics: Additional metrics to track
        """
        if metrics is None:
            metrics = [
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ]

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info(f"Model compiled with learning rate={learning_rate}")

    def create_callbacks(
        self,
        patience: int = 10,
        monitor: str = "val_loss",
        save_best_only: bool = True,
    ) -> list:
        """
        Create training callbacks.

        Args:
            patience: Patience for early stopping
            monitor: Metric to monitor
            save_best_only: Save only best model

        Returns:
            List of callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )

        # Model checkpoint
        checkpoint_path = (
            self.model_dir / f"best_model_{timestamp}.keras"
        )
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor=monitor,
            save_best_only=save_best_only,
            verbose=1,
        )

        # Learning rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        )

        # Tensorboard logging
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=str(self.log_dir / f"logs_{timestamp}"),
            histogram_freq=1,
            write_graph=True,
        )

        callbacks = [
            early_stopping,
            model_checkpoint,
            reduce_lr,
            tensorboard_callback,
        ]

        logger.info("Callbacks created successfully")
        return callbacks

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        learning_rate: float = 0.001,
    ) -> Dict:
        """
        Train the model.

        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Patience for early stopping
            learning_rate: Learning rate

        Returns:
            Training history dictionary
        """
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Compile model
        self.compile_model(learning_rate=learning_rate)

        # Create callbacks
        callbacks = self.create_callbacks(patience=patience)

        logger.info(
            f"Starting training:\n"
            f"  Train samples: {len(X_train)}\n"
            f"  Val samples: {len(X_val)}\n"
            f"  Epochs: {epochs}\n"
            f"  Batch size: {batch_size}"
        )

        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Store history
        self.training_history = {
            "train_loss": history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", []),
            "train_accuracy": history.history.get("accuracy", []),
            "val_accuracy": history.history.get("val_accuracy", []),
        }

        logger.info("Training completed")
        return self.training_history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            threshold: Classification threshold

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X_test)} samples")

        # Get predictions
        y_pred_proba = self.model.predict(X_test, batch_size=batch_size, verbose=0)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            y_test, y_pred_proba, threshold
        )

        # Print metrics
        self.metrics_calculator.print_metrics(metrics, stage="Test")

        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict:
        """
        Perform k-fold cross-validation.

        Args:
            X: Features
            y: Labels
            n_splits: Number of folds
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Dictionary with metrics for each fold
        """
        from sklearn.model_selection import KFold

        logger.info(f"Starting {n_splits}-fold cross-validation")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        fold_metrics_list = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold}/{n_splits}")
            logger.info(f"{'='*50}")

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Recreate and compile model for each fold
            from src.models import HybridDeepfakeDetector

            model = HybridDeepfakeDetector(input_shape=X.shape[1:])
            self.model = model

            # Train
            self.compile_model(learning_rate=learning_rate)
            callbacks = self.create_callbacks(patience=5)

            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            # Evaluate
            y_val_pred = self.model.predict(X_val, verbose=0)
            metrics = self.metrics_calculator.calculate_metrics(y_val, y_val_pred)

            fold_results.append(
                {
                    "fold": fold,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "metrics": metrics,
                }
            )
            fold_metrics_list.append(metrics)

            self.metrics_calculator.print_metrics(metrics, stage=f"Fold {fold}")

        # Calculate average metrics
        avg_metrics = self._average_metrics(fold_metrics_list)

        logger.info(f"\n{'='*50}")
        logger.info("Cross-Validation Summary")
        logger.info(f"{'='*50}")
        self.metrics_calculator.print_metrics(avg_metrics, stage="Average (CV)")

        return {
            "fold_results": fold_results,
            "average_metrics": avg_metrics,
        }

    def save_model(self, path: str = None) -> str:
        """
        Save trained model.

        Args:
            path: Custom path to save model (optional)

        Returns:
            Path where model was saved
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self.model_dir / f"model_{timestamp}.keras")

        self.model.save(path)
        logger.info(f"Model saved to {path}")
        return path

    def load_model(self, path: str):
        """
        Load pre-trained model.

        Args:
            path: Path to saved model
        """
        self.model = keras.models.load_model(path, custom_objects={
            'MultiHeadAttention': None  # Will be imported from custom objects if needed
        })
        logger.info(f"Model loaded from {path}")

    def save_training_history(self, path: str = None):
        """
        Save training history as JSON.

        Args:
            path: Path to save history
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self.log_dir / f"history_{timestamp}.json")

        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {}
        for key, values in self.training_history.items():
            history_serializable[key] = [float(v) for v in values]

        with open(path, "w") as f:
            json.dump(history_serializable, f, indent=2)

        logger.info(f"Training history saved to {path}")
        return path

    @staticmethod
    def _average_metrics(metrics_list: list) -> Dict:
        """Calculate average metrics across folds."""
        avg_metrics = {}
        metric_keys = metrics_list[0].keys()

        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
            if values:
                avg_metrics[key] = np.mean(values)

        return avg_metrics
