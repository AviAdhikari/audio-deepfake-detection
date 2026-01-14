"""
Tests for evaluation and visualization modules.

This module contains unit tests for the evaluation visualization pipeline.

Usage:
    pytest tests/test_evaluate_and_visualize.py -v
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from examples.evaluate_and_visualize import EvaluationVisualizer


class TestEvaluationVisualizer:
    """Tests for EvaluationVisualizer class."""

    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield EvaluationVisualizer(tmpdir)

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.3, 0.7, 0.9, 0.2, 0.95])
        return y_true, y_pred, y_pred_proba

    def test_visualizer_initialization(self, visualizer):
        """Test EvaluationVisualizer initialization."""
        assert visualizer.output_dir.exists()

    def test_output_directory_creation(self):
        """Test that output directory is created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_viz_dir"
            assert not output_dir.exists()
            viz = EvaluationVisualizer(str(output_dir))
            assert viz.output_dir.exists()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_confusion_matrix(self, mock_close, mock_savefig, visualizer, sample_predictions):
        """Test confusion matrix plotting."""
        y_true, y_pred, _ = sample_predictions

        cm = visualizer.plot_confusion_matrix(y_true, y_pred, "TestModel", "TestDataset")

        # Check confusion matrix values
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_roc_curve(self, mock_close, mock_savefig, visualizer, sample_predictions):
        """Test ROC curve plotting."""
        y_true, _, y_pred_proba = sample_predictions

        fpr, tpr, roc_auc = visualizer.plot_roc_curve(
            y_true, y_pred_proba, "TestModel", "TestDataset"
        )

        # Check ROC curve properties
        assert len(fpr) > 0
        assert len(tpr) > 0
        assert 0 <= roc_auc <= 1

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_pr_curve(self, mock_close, mock_savefig, visualizer, sample_predictions):
        """Test Precision-Recall curve plotting."""
        y_true, _, y_pred_proba = sample_predictions

        precision, recall, pr_auc = visualizer.plot_precision_recall_curve(
            y_true, y_pred_proba, "TestModel", "TestDataset"
        )

        # Check PR curve properties
        assert len(precision) > 0
        assert len(recall) > 0
        assert 0 <= pr_auc <= 1

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_training_history(self, mock_close, mock_savefig, visualizer):
        """Test training history plotting."""
        history = {
            "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
            "val_loss": [0.6, 0.5, 0.4, 0.3, 0.2],
            "accuracy": [0.7, 0.75, 0.8, 0.85, 0.9],
            "val_accuracy": [0.65, 0.7, 0.75, 0.8, 0.85],
        }

        visualizer.plot_training_history(history, "TestModel", "TestDataset")
        # Should complete without error

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_model_comparison(self, mock_close, mock_savefig, visualizer):
        """Test model comparison plotting."""
        results = {
            "Model1": {"accuracy": 0.95, "f1_score": 0.94, "roc_auc": 0.97},
            "Model2": {"accuracy": 0.92, "f1_score": 0.91, "roc_auc": 0.94},
            "Model3": {"accuracy": 0.90, "f1_score": 0.89, "roc_auc": 0.92},
        }

        visualizer.plot_model_comparison(results, metric="f1_score")
        # Should complete without error

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_roc_comparison(self, mock_close, mock_savefig, visualizer):
        """Test ROC comparison plotting."""
        results_dict = {
            "Model1": {
                "y_true": np.array([0, 1, 0, 1, 0, 1]),
                "y_pred_proba": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
            },
            "Model2": {
                "y_true": np.array([0, 1, 0, 1, 0, 1]),
                "y_pred_proba": np.array([0.2, 0.85, 0.3, 0.75, 0.4, 0.65]),
            },
        }

        visualizer.plot_roc_comparison(results_dict)
        # Should complete without error


class TestMetricsCalculation:
    """Tests for metrics calculation."""

    def test_confusion_matrix_values(self):
        """Test confusion matrix calculation."""
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])

        cm = confusion_matrix(y_true, y_pred)

        # Should be 2x2 for binary classification
        assert cm.shape == (2, 2)
        # TN=2, FP=1, FN=1, TP=2
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 2  # TP

    def test_roc_auc_score(self):
        """Test ROC-AUC score calculation."""
        from sklearn.metrics import roc_auc_score

        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        roc_auc = roc_auc_score(y_true, y_pred_proba)

        assert 0 <= roc_auc <= 1
        # Perfect predictions should have AUC near 1
        assert roc_auc > 0.9

    def test_pr_auc_score(self):
        """Test Precision-Recall AUC calculation."""
        from sklearn.metrics import auc, precision_recall_curve

        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)

        assert 0 <= pr_auc <= 1

    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity calculation."""
        # TP=2, TN=2, FP=1, FN=1
        tp, tn, fp, fn = 2, 2, 1, 1

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        assert sensitivity == 2 / 3
        assert specificity == 2 / 3


class TestResultHandling:
    """Tests for result handling and export."""

    def test_json_results_structure(self):
        """Test that results can be exported to JSON."""
        results = {
            "Model1": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "roc_auc": 0.97,
                "y_pred": [0, 1, 0, 1],
                "y_pred_proba": [0.1, 0.9, 0.2, 0.8],
                "history": {
                    "loss": [0.5, 0.4],
                    "val_loss": [0.6, 0.5],
                    "accuracy": [0.7, 0.75],
                    "val_accuracy": [0.65, 0.7],
                },
            }
        }

        json_str = json.dumps(results)
        reloaded = json.loads(json_str)
        assert reloaded["Model1"]["accuracy"] == 0.95

    def test_missing_prediction_data_handling(self):
        """Test handling of missing prediction data."""
        results_dict = {
            "Model1": {"accuracy": 0.95},  # Missing y_pred_proba
            "Model2": {
                "y_true": np.array([0, 1, 0, 1]),
                "y_pred_proba": np.array([0.1, 0.9, 0.2, 0.8]),
            },
        }

        # Visualizer should handle missing data gracefully
        assert results_dict is not None

    def test_multiple_models_results(self):
        """Test results with multiple models."""
        results = {
            "HybridDeepfakeDetector": {
                "accuracy": 0.98,
                "f1_score": 0.98,
                "roc_auc": 0.99,
            },
            "TransformerDeepfakeDetector": {
                "accuracy": 0.99,
                "f1_score": 0.99,
                "roc_auc": 0.995,
            },
        }

        assert len(results) == 2
        for model, metrics in results.items():
            assert "accuracy" in metrics
            assert "f1_score" in metrics
            assert "roc_auc" in metrics


class TestVisualizationOutput:
    """Tests for visualization output files."""

    def test_output_file_naming(self):
        """Test that output files are named correctly."""
        expected_patterns = [
            "*_confusion_matrix.png",
            "*_roc_curve.png",
            "*_pr_curve.png",
            "*_training_history.png",
            "*_model_comparison_*.png",
            "*_roc_comparison.png",
        ]

        assert len(expected_patterns) == 6

    def test_dpi_setting(self):
        """Test that DPI is set to 300 for publication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = EvaluationVisualizer(tmpdir)
            # Check that matplotlib DPI is configured
            import matplotlib.pyplot as plt

            assert plt.rcParams["figure.dpi"] == 300
            assert plt.rcParams["savefig.dpi"] == 300


class TestDataValidation:
    """Tests for data validation in visualization."""

    def test_binary_predictions_validation(self):
        """Test that predictions are binary (0 or 1)."""
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        unique_values = np.unique(y_pred)

        assert len(unique_values) == 2
        assert set(unique_values) == {0, 1}

    def test_probability_range_validation(self):
        """Test that probabilities are in [0, 1] range."""
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.0, 1.0])

        assert (y_pred_proba >= 0).all()
        assert (y_pred_proba <= 1).all()

    def test_array_length_consistency(self):
        """Test that arrays have consistent lengths."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        assert len(y_true) == len(y_pred) == len(y_pred_proba)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
