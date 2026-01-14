"""
Pytest configuration and fixtures for audio deepfake detection tests.

This module provides shared fixtures and configuration for all tests.

Usage:
    pytest --fixtures  (to see all available fixtures)
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary test data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio():
    """Create sample audio data."""
    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    # Generate a simple sine wave
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_features():
    """Create sample MFCC features."""
    n_samples = 10
    n_mfcc = 13
    n_frames = 256

    # MFCC features
    mfcc = np.random.randn(n_samples, n_mfcc, n_frames).astype(np.float32)

    # Delta features
    delta = np.random.randn(n_samples, n_mfcc, n_frames).astype(np.float32)

    # Stack into feature matrix
    features = np.stack([mfcc, delta], axis=1)  # (N, 2, 13, 256)

    return features


@pytest.fixture
def sample_labels():
    """Create sample binary labels."""
    return np.random.randint(0, 2, 10).astype(np.float32)


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.3, 0.7, 0.9, 0.2, 0.95])
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def sample_training_history():
    """Create sample training history."""
    return {
        "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
        "val_loss": [0.6, 0.5, 0.4, 0.3, 0.2],
        "accuracy": [0.7, 0.75, 0.8, 0.85, 0.9],
        "val_accuracy": [0.65, 0.7, 0.75, 0.8, 0.85],
    }


@pytest.fixture
def sample_results():
    """Create sample results dictionary."""
    return {
        "HybridDeepfakeDetector": {
            "accuracy": 0.9823,
            "precision": 0.9815,
            "recall": 0.9831,
            "f1_score": 0.9823,
            "roc_auc": 0.9923,
            "y_pred": [0, 1, 0, 1, 0, 1],
            "y_pred_proba": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            "history": {
                "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
                "val_loss": [0.6, 0.5, 0.4, 0.3, 0.2],
                "accuracy": [0.7, 0.75, 0.8, 0.85, 0.9],
                "val_accuracy": [0.65, 0.7, 0.75, 0.8, 0.85],
            },
        },
        "TransformerDeepfakeDetector": {
            "accuracy": 0.9910,
            "precision": 0.9905,
            "recall": 0.9915,
            "f1_score": 0.9910,
            "roc_auc": 0.9960,
            "y_pred": [0, 1, 0, 1, 0, 1],
            "y_pred_proba": [0.05, 0.95, 0.15, 0.85, 0.25, 0.75],
            "history": {
                "loss": [0.4, 0.3, 0.2, 0.1, 0.05],
                "val_loss": [0.5, 0.4, 0.3, 0.2, 0.15],
                "accuracy": [0.75, 0.8, 0.85, 0.9, 0.95],
                "val_accuracy": [0.7, 0.75, 0.8, 0.85, 0.9],
            },
        },
    }


@pytest.fixture
def temp_results_dir():
    """Create a temporary results directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        yield results_dir


@pytest.fixture
def temp_models_dir():
    """Create a temporary models directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        yield models_dir


@pytest.fixture
def temp_visualizations_dir():
    """Create a temporary visualizations directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        viz_dir = Path(tmpdir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        yield viz_dir


# Pytest configuration options
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests containing 'integration' as integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # Mark tests not containing 'integration' as unit tests
        else:
            item.add_marker(pytest.mark.unit)


# Fixtures for mocking and testing
@pytest.fixture
def mock_model():
    """Create a mock model."""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([0.1, 0.9, 0.2, 0.8]))
    model.summary = MagicMock()
    return model


@pytest.fixture
def mock_trainer():
    """Create a mock trainer."""
    from unittest.mock import MagicMock

    trainer = MagicMock()
    trainer.train = MagicMock(
        return_value={
            "loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "accuracy": [0.7, 0.75, 0.8],
            "val_accuracy": [0.65, 0.7, 0.75],
        }
    )
    return trainer


# Test data constants
TEST_AUDIO_DURATION = 1.0  # 1 second
TEST_AUDIO_SR = 16000  # 16 kHz
TEST_N_MFCC = 13
TEST_N_FRAMES = 256
TEST_FEATURE_SHAPE = (2, TEST_N_MFCC, TEST_N_FRAMES)
TEST_N_SAMPLES = 10


# Logging configuration for tests
def pytest_configure(config):
    """Configure logging for tests."""
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
