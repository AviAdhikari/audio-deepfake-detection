"""Metrics calculation for model evaluation."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for binary classification."""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5):
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int).flatten()
        y_true_flat = y_true.flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_true_flat, y_pred)
        precision = precision_score(y_true_flat, y_pred, zero_division=0)
        recall = recall_score(y_true_flat, y_pred, zero_division=0)
        f1 = f1_score(y_true_flat, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true_flat, y_pred_proba.flatten())

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

        return metrics

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Find optimal threshold based on F1-score.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities

        Returns:
            Tuple of (optimal_threshold, best_f1_score)
        """
        fpr, tpr, thresholds = roc_curve(y_true.flatten(), y_pred_proba.flatten())

        best_f1 = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int).flatten()
            f1 = f1_score(y_true.flatten(), y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        logger.info(f"Optimal threshold: {best_threshold:.4f}, F1-score: {best_f1:.4f}")
        return best_threshold, best_f1

    @staticmethod
    def print_metrics(metrics: dict, stage: str = "Validation"):
        """Print metrics in a formatted way."""
        logger.info(f"\n{'='*50}")
        logger.info(f"{stage} Metrics")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy:    {metrics['accuracy']:.4f}")
        logger.info(f"Precision:   {metrics['precision']:.4f}")
        logger.info(f"Recall:      {metrics['recall']:.4f}")
        logger.info(f"F1-Score:    {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
        logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"TP: {metrics['tp']}, TN: {metrics['tn']}, "
                   f"FP: {metrics['fp']}, FN: {metrics['fn']}")
        logger.info(f"{'='*50}\n")
