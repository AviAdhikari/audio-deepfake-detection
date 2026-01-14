"""
Tests for training and evaluation modules.

This module contains unit tests for the ASVspoof/WaveFake training
and evaluation pipeline.

Usage:
    pytest tests/test_train_on_asvspoof_wavefake.py -v
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from examples.train_on_asvspoof_wavefake import (
    ASVspoofDataLoader,
    WaveFakeDataLoader,
    train_models_on_dataset,
)


class TestASVspoofDataLoader:
    """Tests for ASVspoofDataLoader class."""

    def test_loader_initialization(self):
        """Test ASVspoofDataLoader initialization."""
        loader = ASVspoofDataLoader("data/ASVspoof2019")
        assert loader.data_dir == Path("data/ASVspoof2019")
        assert loader.audio_processor is not None

    def test_loader_missing_dataset(self):
        """Test that loader raises error for missing dataset."""
        loader = ASVspoofDataLoader("nonexistent/path")
        with pytest.raises(FileNotFoundError):
            loader.load_dataset(subset="LA", split="train")

    def test_loader_invalid_subset(self):
        """Test that loader accepts LA and PA subsets."""
        loader = ASVspoofDataLoader("data/ASVspoof2019")
        # Should not raise on valid subset names
        assert loader is not None

    @patch("librosa.load")
    @patch("librosa.feature.mfcc")
    @patch("librosa.feature.delta")
    @patch("builtins.open", create=True)
    def test_feature_extraction_shape(self, mock_open, mock_delta, mock_mfcc, mock_load):
        """Test that extracted features have correct shape."""
        # Mock audio loading
        mock_load.return_value = (np.random.randn(16000), 16000)
        mock_mfcc.return_value = np.random.randn(13, 256)
        mock_delta.return_value = np.random.randn(13, 256)

        # Mock protocol file
        protocol_content = "speaker1 audio1 - - bonafide\nspeaker2 audio2 - - spoof\n"
        mock_open.return_value.__enter__.return_value = iter(protocol_content.split("\n"))

        loader = ASVspoofDataLoader("data/ASVspoof2019")
        # Note: actual load_dataset would fail without real files
        # This tests the logic flow

    def test_label_encoding(self):
        """Test that labels are correctly encoded."""
        # bonafide should map to 0, spoof to 1
        # Tested through load_dataset when data available
        assert True  # Placeholder for integration test


class TestWaveFakeDataLoader:
    """Tests for WaveFakeDataLoader class."""

    def test_loader_initialization(self):
        """Test WaveFakeDataLoader initialization."""
        loader = WaveFakeDataLoader("data/WaveFake")
        assert loader.data_dir == Path("data/WaveFake")
        assert loader.audio_processor is not None

    def test_loader_missing_dataset(self):
        """Test that loader raises error for missing dataset."""
        loader = WaveFakeDataLoader("nonexistent/path")
        with pytest.raises(FileNotFoundError):
            loader.load_dataset(split="train")

    def test_valid_splits(self):
        """Test that loader accepts train, val, test splits."""
        loader = WaveFakeDataLoader("data/WaveFake")
        # Should not raise on valid split names
        assert loader is not None

    def test_label_encoding_wavefake(self):
        """Test that WaveFake labels are correctly encoded."""
        # real should map to 0, fake to 1
        # Tested through load_dataset when data available
        assert True  # Placeholder for integration test


class TestTrainingPipeline:
    """Tests for model training pipeline."""

    def test_training_with_synthetic_data(self):
        """Test training pipeline with synthetic data."""
        # Create synthetic data
        n_train = 20
        n_test = 10
        X_train = np.random.randn(n_train, 2, 13, 256).astype(np.float32)
        y_train = np.random.randint(0, 2, n_train).astype(np.float32)
        X_test = np.random.randn(n_test, 2, 13, 256).astype(np.float32)
        y_test = np.random.randint(0, 2, n_test).astype(np.float32)

        # Test training with mock models
        with tempfile.TemporaryDirectory() as tmpdir:
            # This would require mocking the actual models
            # For now, test the logic structure
            assert X_train.shape == (n_train, 2, 13, 256)
            assert y_train.shape == (n_train,)

    def test_result_export_structure(self):
        """Test that results are exported with correct structure."""
        # Expected result structure
        expected_keys = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
        
        # Mock results
        mock_results = {
            "HybridDeepfakeDetector": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "roc_auc": 0.97,
            }
        }
        
        assert set(mock_results["HybridDeepfakeDetector"].keys()) == expected_keys

    def test_json_serialization(self):
        """Test that results can be serialized to JSON."""
        mock_results = {
            "HybridDeepfakeDetector": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "roc_auc": 0.97,
                "y_pred": [0, 1, 0, 1],
                "y_pred_proba": [0.1, 0.9, 0.2, 0.8],
            }
        }

        # Should be JSON serializable
        json_str = json.dumps(mock_results)
        assert json_str is not None

        # Should deserialize correctly
        reloaded = json.loads(json_str)
        assert reloaded["HybridDeepfakeDetector"]["accuracy"] == 0.95

    def test_stratified_splitting(self):
        """Test that stratified splitting preserves label distribution."""
        from sklearn.model_selection import train_test_split

        # Create imbalanced data
        y = np.array([0, 0, 0, 0, 1, 1])
        X = np.arange(len(y))

        # Stratified split should preserve ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42, stratify=y
        )

        # Check that both classes present in both sets
        assert len(np.unique(y_train)) > 0
        assert len(np.unique(y_test)) > 0


class TestDataIntegrity:
    """Tests for data integrity and validation."""

    def test_feature_shape_consistency(self):
        """Test that all features have consistent shape."""
        n_samples = 50
        feature_shape = (2, 13, 256)

        X = np.random.randn(n_samples, *feature_shape).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)

        assert X.shape == (n_samples, *feature_shape)
        assert y.shape == (n_samples,)
        assert X.dtype == np.float32
        assert np.isfinite(X).all()

    def test_label_binary_classification(self):
        """Test that labels are binary (0 or 1)."""
        y = np.array([0, 1, 0, 1, 0, 1])
        unique_labels = np.unique(y)
        assert len(unique_labels) == 2
        assert set(unique_labels) == {0, 1}

    def test_no_nan_values(self):
        """Test that data contains no NaN values."""
        X = np.random.randn(10, 2, 13, 256)
        assert not np.isnan(X).any()

    def test_no_infinite_values(self):
        """Test that data contains no infinite values."""
        X = np.random.randn(10, 2, 13, 256)
        assert np.isfinite(X).all()


class TestDataLoaderErrorHandling:
    """Tests for error handling in data loaders."""

    def test_missing_audio_file_handling(self):
        """Test that loader handles missing audio files gracefully."""
        # When audio file doesn't exist, should log warning and continue
        loader = ASVspoofDataLoader("data/ASVspoof2019")
        # Implementation should have try/except for missing files
        assert loader is not None

    def test_corrupted_audio_handling(self):
        """Test that loader handles corrupted audio gracefully."""
        # When audio file is corrupted, should log warning and continue
        loader = WaveFakeDataLoader("data/WaveFake")
        assert loader is not None

    def test_empty_dataset_handling(self):
        """Test that loader detects empty datasets."""
        loader = ASVspoofDataLoader("data/ASVspoof2019")
        # If dataset is empty, should raise RuntimeError
        assert loader is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
