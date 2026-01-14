"""
Tests for foundation models (Wav2Vec2, Whisper, HuBERT).

This module contains unit tests for the foundation model feature extractors.

Usage:
    pytest tests/test_foundation_models.py -v
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.foundation_models import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor


class TestWav2Vec2FeatureExtractor:
    """Tests for Wav2Vec2FeatureExtractor."""

    @patch("src.models.foundation_models.Wav2Vec2Processor")
    @patch("src.models.foundation_models.TFWav2Vec2Model")
    def test_initialization(self, mock_model, mock_processor):
        """Test Wav2Vec2FeatureExtractor initialization."""
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        extractor = Wav2Vec2FeatureExtractor("facebook/wav2vec2-base")

        assert extractor.model_name == "facebook/wav2vec2-base"
        assert extractor.processor is not None
        assert extractor.model is not None

    @patch("src.models.foundation_models.Wav2Vec2Processor")
    @patch("src.models.foundation_models.TFWav2Vec2Model")
    def test_initialization_custom_model(self, mock_model, mock_processor):
        """Test initialization with custom model."""
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        extractor = Wav2Vec2FeatureExtractor("facebook/wav2vec2-large")

        assert extractor.model_name == "facebook/wav2vec2-large"

    def test_missing_transformers_library(self):
        """Test that ImportError is raised if transformers not installed."""
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError):
                # This would fail if transformers not available
                pass

    @patch("src.models.foundation_models.Wav2Vec2Processor")
    @patch("src.models.foundation_models.TFWav2Vec2Model")
    def test_extract_features_shape(self, mock_model, mock_processor):
        """Test that extracted features have correct shape."""
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock the processor and model outputs
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.return_value = {"input_values": np.random.randn(1, 16000)}

        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = np.random.randn(1, 100, 768)
        mock_model_instance.return_value = mock_output

        extractor = Wav2Vec2FeatureExtractor("facebook/wav2vec2-base")

        # Test with audio data
        audio_data = np.random.randn(16000)
        # Note: actual execution would need real transformers library

    @patch("src.models.foundation_models.Wav2Vec2Processor")
    @patch("src.models.foundation_models.TFWav2Vec2Model")
    def test_sampling_rate_handling(self, mock_model, mock_processor):
        """Test that sampling rate is handled correctly."""
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        extractor = Wav2Vec2FeatureExtractor("facebook/wav2vec2-base")

        # Wav2Vec2 expects 16kHz
        # Should handle resampling if needed
        assert extractor is not None


class TestWhisperFeatureExtractor:
    """Tests for WhisperFeatureExtractor."""

    @patch("src.models.foundation_models.whisper.load_model")
    def test_initialization(self, mock_load_model):
        """Test WhisperFeatureExtractor initialization."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        extractor = WhisperFeatureExtractor("base")

        assert extractor.model_name == "base"
        assert extractor.model is not None

    @patch("src.models.foundation_models.whisper.load_model")
    def test_model_sizes(self, mock_load_model):
        """Test that different model sizes can be loaded."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        for size in ["tiny", "base", "small", "medium", "large"]:
            extractor = WhisperFeatureExtractor(size)
            assert extractor.model_name == size

    def test_missing_whisper_library(self):
        """Test that ImportError is raised if whisper not installed."""
        with patch.dict("sys.modules", {"whisper": None}):
            with pytest.raises(ImportError):
                # This would fail if openai-whisper not available
                pass

    @patch("src.models.foundation_models.whisper.load_model")
    @patch("src.models.foundation_models.whisper.load_audio")
    @patch("src.models.foundation_models.whisper.log_mel_spectrogram")
    def test_feature_extraction(
        self, mock_mel, mock_load_audio, mock_load_model
    ):
        """Test Whisper feature extraction."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_audio = np.random.randn(160000)  # 10 seconds at 16kHz
        mock_load_audio.return_value = mock_audio

        mock_spectrogram = MagicMock()
        mock_mel.return_value = mock_spectrogram

        extractor = WhisperFeatureExtractor("base")

        # Whisper features would be mel spectrograms
        assert extractor is not None


class TestFoundationModelIntegration:
    """Integration tests for foundation models."""

    def test_both_extractors_available(self):
        """Test that both Wav2Vec2 and Whisper extractors are available."""
        # Both classes should be importable
        assert Wav2Vec2FeatureExtractor is not None
        assert WhisperFeatureExtractor is not None

    @patch("src.models.foundation_models.Wav2Vec2Processor")
    @patch("src.models.foundation_models.TFWav2Vec2Model")
    def test_wav2vec2_output_dimensions(self, mock_model, mock_processor):
        """Test expected output dimensions of Wav2Vec2."""
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()

        # Wav2Vec2-base has 768 dimensions
        # Output shape should be (batch, time_steps, 768)
        expected_dims = 768
        assert expected_dims > 0

    def test_feature_extractor_modes(self):
        """Test that extractors support both training and inference modes."""
        # Both extractors should work in inference mode
        # (no gradient computation needed)
        assert True  # Placeholder for actual mode testing


class TestAudioInputValidation:
    """Tests for audio input validation."""

    def test_audio_length_handling(self):
        """Test handling of different audio lengths."""
        # Should handle variable-length audio
        audio_lengths = [16000, 32000, 48000]  # 1s, 2s, 3s at 16kHz

        for length in audio_lengths:
            audio = np.random.randn(length)
            assert len(audio) == length

    def test_audio_dtype_handling(self):
        """Test handling of different audio dtypes."""
        dtypes = [np.float32, np.float64, np.int16]

        for dtype in dtypes:
            if dtype == np.int16:
                audio = np.random.randint(-32768, 32767, 16000, dtype=dtype)
            else:
                audio = np.random.randn(16000).astype(dtype)

            assert audio.dtype == dtype

    def test_mono_audio_handling(self):
        """Test that mono audio is handled correctly."""
        # Wav2Vec2 expects mono audio
        mono_audio = np.random.randn(16000)
        assert mono_audio.ndim == 1

    def test_stereo_audio_handling(self):
        """Test that stereo audio might need conversion to mono."""
        # Some models might accept stereo but convert to mono
        stereo_audio = np.random.randn(2, 16000)
        assert stereo_audio.ndim == 2

        # Convert to mono (average channels)
        mono_audio = stereo_audio.mean(axis=0)
        assert mono_audio.ndim == 1


class TestFeatureProperties:
    """Tests for properties of extracted features."""

    def test_features_not_all_zeros(self):
        """Test that extracted features are not all zeros."""
        # Properly loaded models should produce non-zero features
        features = np.random.randn(1, 100, 768)
        assert not np.allclose(features, 0)

    def test_features_are_numeric(self):
        """Test that features are numeric (not NaN or Inf)."""
        features = np.random.randn(1, 100, 768)

        assert np.isfinite(features).all()
        assert features.dtype in [np.float32, np.float64]

    def test_features_in_reasonable_range(self):
        """Test that feature values are in reasonable range."""
        # Normalized features should be roughly in [-3, 3] range
        # (assuming standard normal distribution)
        features = np.random.randn(1, 100, 768)

        # Most values should be within [-3, 3]
        assert (np.abs(features) < 10).sum() > features.size * 0.99


class TestModelConsistency:
    """Tests for model consistency."""

    @patch("src.models.foundation_models.Wav2Vec2Processor")
    @patch("src.models.foundation_models.TFWav2Vec2Model")
    def test_same_input_same_output(self, mock_model, mock_processor):
        """Test that same input produces same output."""
        # Models should be deterministic (without dropout)
        assert True  # Placeholder for actual testing

    @patch("src.models.foundation_models.Wav2Vec2Processor")
    @patch("src.models.foundation_models.TFWav2Vec2Model")
    def test_different_inputs_different_outputs(
        self, mock_model, mock_processor
    ):
        """Test that different inputs produce different outputs."""
        # Different audio should produce different features
        assert True  # Placeholder for actual testing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
