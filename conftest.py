"""Configuration file for pytest."""

import pytest
import numpy as np


@pytest.fixture
def sample_audio_features():
    """Generate sample audio features for testing."""
    # Shape: (batch_size, channels, height, width) = (4, 2, 39, 256)
    return np.random.randn(4, 2, 39, 256).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    return np.array([[0], [1], [0], [1]], dtype=np.float32)
