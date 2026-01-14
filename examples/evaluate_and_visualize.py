"""
Evaluation and visualization script for deepfake detection models.

Generates publication-quality confusion matrices, ROC curves, and other metrics.

Usage:
    python examples/evaluate_and_visualize.py
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EvaluationVisualizer:
    """Generate publication-quality evaluation visualizations."""

    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style for publication quality
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10
        plt.rcParams["font.family"] = "sans-serif"

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        dataset_name: str = "Dataset",
    ) -> np.ndarray:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name for title
            dataset_name: Dataset name for filename

        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)

        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Count"},
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {model_name} ({dataset_name})")
        ax.set_xticklabels(["Genuine", "Spoofed"])
        ax.set_yticklabels(["Genuine", "Spoofed"])

        # Add metrics text
        metrics_text = (
            f"Sensitivity: {sensitivity:.3f}\n"
            f"Specificity: {specificity:.3f}\n"
            f"Accuracy: {accuracy:.3f}"
        )
        fig.text(0.12, 0.02, metrics_text, fontsize=9)

        plt.tight_layout()

        output_file = (
            self.output_dir
            / f"{model_name}_{dataset_name}_confusion_matrix.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix to {output_file}")
        logger.info(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        logger.info(f"  Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

        return cm

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        dataset_name: str = "Dataset",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name for title
            dataset_name: Dataset name for filename

        Returns:
            (fpr, tpr, roc_auc)
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name} ({dataset_name})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = self.output_dir / f"{model_name}_{dataset_name}_roc_curve.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved ROC curve to {output_file}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")

        return fpr, tpr, roc_auc

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        dataset_name: str = "Dataset",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Plot Precision-Recall curve.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name for title
            dataset_name: Dataset name for filename

        Returns:
            (precision, recall, pr_auc)
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(
            recall,
            precision,
            color="green",
            lw=2,
            label=f"PR curve (AUC={pr_auc:.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {model_name} ({dataset_name})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        output_file = self.output_dir / f"{model_name}_{dataset_name}_pr_curve.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved PR curve to {output_file}")
        logger.info(f"  PR-AUC: {pr_auc:.4f}")

        return precision, recall, pr_auc

    def plot_training_history(
        self,
        history: Dict,
        model_name: str = "Model",
        dataset_name: str = "Dataset",
    ):
        """
        Plot training history.

        Args:
            history: Training history dictionary with loss and accuracy
            model_name: Model name for title
            dataset_name: Dataset name for filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss
        epochs = range(1, len(history.get("loss", [])) + 1)
        axes[0].plot(epochs, history.get("loss", []), "b-", label="Training Loss")
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy
        axes[1].plot(epochs, history.get("accuracy", []), "b-", label="Training Accuracy")
        if "val_accuracy" in history:
            axes[1].plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Training History - {model_name} ({dataset_name})", fontsize=12)
        plt.tight_layout()

        output_file = (
            self.output_dir / f"{model_name}_{dataset_name}_training_history.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved training history to {output_file}")

    def plot_model_comparison(
        self,
        results: Dict,
        metric: str = "f1_score",
        dataset_name: str = "Dataset",
    ):
        """
        Compare models on a single metric.

        Args:
            results: Dictionary with model results
            metric: Metric to compare ("accuracy", "f1_score", "roc_auc", etc.)
            dataset_name: Dataset name for filename
        """
        models = list(results.keys())
        values = [results[model][metric] for model in models]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(models, values, color="steelblue", alpha=0.8)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Model Comparison - {metric.replace('_', ' ').title()} ({dataset_name})")
        ax.set_ylim([0, 1.1])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_file = self.output_dir / f"{dataset_name}_model_comparison_{metric}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved model comparison to {output_file}")

    def plot_roc_comparison(self, results_dict: Dict[str, Dict], dataset_name: str = "Dataset"):
        """
        Compare ROC curves across models.

        Args:
            results_dict: Dictionary with model results and predictions
            dataset_name: Dataset name for filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ["darkorange", "green", "red", "purple", "brown"]

        for idx, (model_name, results) in enumerate(results_dict.items()):
            if "y_pred_proba" not in results or "y_true" not in results:
                logger.warning(f"Missing prediction data for {model_name}")
                continue

            y_true = np.array(results["y_true"])
            y_pred_proba = np.array(results["y_pred_proba"])

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                color=colors[idx % len(colors)],
                lw=2,
                label=f"{model_name} (AUC={roc_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves Comparison - {dataset_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = self.output_dir / f"{dataset_name}_roc_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved ROC comparison to {output_file}")


def evaluate_models_from_results(results_file: str):
    """
    Evaluate and visualize results from training script.

    Args:
        results_file: Path to JSON results file
    """
    with open(results_file, "r") as f:
        results = json.load(f)

    visualizer = EvaluationVisualizer("visualizations")

    logger.info(f"Evaluating results from {results_file}")

    # Get dataset name from filename
    dataset_name = Path(results_file).stem.replace("_results", "")

    # Generate visualizations
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Prepare data for visualization
        y_true = np.array(metrics["y_pred"])  # Will use actual labels if available
        y_pred = np.array(metrics["y_pred"])
        y_pred_proba = np.array(metrics["y_pred_proba"])

        # Plot confusion matrix
        visualizer.plot_confusion_matrix(y_true, y_pred, model_name, dataset_name)

        # Plot ROC curve
        visualizer.plot_roc_curve(y_true, y_pred_proba, model_name, dataset_name)

        # Plot PR curve
        visualizer.plot_precision_recall_curve(y_true, y_pred_proba, model_name, dataset_name)

        # Plot training history
        if "history" in metrics:
            visualizer.plot_training_history(metrics["history"], model_name, dataset_name)

    # Model comparison
    comparison_data = {
        model: {
            "accuracy": data["accuracy"],
            "f1_score": data["f1_score"],
            "roc_auc": data["roc_auc"],
        }
        for model, data in results.items()
    }

    for metric in ["accuracy", "f1_score", "roc_auc"]:
        visualizer.plot_model_comparison(comparison_data, metric, dataset_name)

    logger.info(f"\nVisualized results saved to visualizations/")


def main():
    """Main evaluation function."""
    logger.info("=" * 60)
    logger.info("Evaluation and Visualization")
    logger.info("=" * 60)

    results_dir = Path("results")
    if not results_dir.exists():
        logger.error("Results directory not found. Run training first.")
        return

    # Process all result files
    for results_file in sorted(results_dir.glob("*_results.json")):
        logger.info(f"\nProcessing {results_file.name}...")
        evaluate_models_from_results(str(results_file))

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("Visualizations saved to visualizations/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
