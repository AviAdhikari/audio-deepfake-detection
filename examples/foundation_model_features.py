"""Example: Using pre-trained foundation models for feature extraction."""

import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.foundation_models import (
    Wav2Vec2FeatureExtractor,
    HuBERTFeatureExtractor,
    WhisperFeatureExtractor,
    FoundationModelEnsemble,
)
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_synthetic_audio(duration: float = 5.0, sr: int = 16000):
    """Generate synthetic audio for demonstration."""
    logger.info(f"Generating synthetic audio ({duration}s at {sr}Hz)...")

    n_samples = int(sr * duration)
    # Create mixture of sine waves
    t = np.arange(n_samples) / sr
    audio = (
        0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.random.randn(n_samples)
    )

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    logger.info(f"Generated audio shape: {audio.shape}")
    return audio, sr


def example_wav2vec2_extraction():
    """Example: Extract features using Wav2Vec2."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Wav2Vec2 Feature Extraction")
    logger.info("=" * 60)

    try:
        audio, sr = generate_synthetic_audio()

        logger.info("Initializing Wav2Vec2FeatureExtractor...")
        extractor = Wav2Vec2FeatureExtractor()

        logger.info("Extracting features...")
        features = extractor.extract_features(audio, sr=sr)

        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Features dtype: {features.dtype}")
        logger.info(f"Features range: [{features.min():.4f}, {features.max():.4f}]")

    except ImportError as e:
        logger.warning(f"Wav2Vec2 not available: {e}")


def example_hubert_extraction():
    """Example: Extract features using HuBERT."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: HuBERT Feature Extraction")
    logger.info("=" * 60)

    try:
        audio, sr = generate_synthetic_audio()

        logger.info("Initializing HuBERTFeatureExtractor...")
        extractor = HuBERTFeatureExtractor()

        logger.info("Extracting features...")
        features = extractor.extract_features(audio, sr=sr)

        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Features dtype: {features.dtype}")
        logger.info(f"Features range: [{features.min():.4f}, {features.max():.4f}]")

    except ImportError as e:
        logger.warning(f"HuBERT not available: {e}")


def example_whisper_extraction():
    """Example: Extract features using Whisper."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Whisper Feature Extraction")
    logger.info("=" * 60)

    try:
        audio, sr = generate_synthetic_audio()

        logger.info("Initializing WhisperFeatureExtractor...")
        extractor = WhisperFeatureExtractor()

        logger.info("Extracting features...")
        features = extractor.extract_features(audio, sr=sr)

        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Features dtype: {features.dtype}")
        logger.info(f"Features range: [{features.min():.4f}, {features.max():.4f}]")

    except ImportError as e:
        logger.warning(f"Whisper not available: {e}")


def example_foundation_model_ensemble():
    """Example: Use ensemble of foundation models."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Foundation Model Ensemble")
    logger.info("=" * 60)

    try:
        audio, sr = generate_synthetic_audio()

        logger.info("Initializing FoundationModelEnsemble...")
        ensemble = FoundationModelEnsemble(model_names=["wav2vec2", "whisper"])

        logger.info("Extracting features from ensemble...")
        features = ensemble.extract_features(audio, sr=sr)

        logger.info(f"Ensemble features shape: {features.shape}")
        logger.info(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
        logger.info("Ensemble combines multiple models for better generalization")

    except Exception as e:
        logger.warning(f"Ensemble extraction issue: {e}")


def example_feature_comparison():
    """Example: Compare features from different extractors."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Feature Comparison")
    logger.info("=" * 60)

    audio, sr = generate_synthetic_audio(duration=2.0)

    extractors = {
        "Wav2Vec2": Wav2Vec2FeatureExtractor(),
        "HuBERT": HuBERTFeatureExtractor(),
        "Whisper": WhisperFeatureExtractor(),
    }

    logger.info("Extracting features from all models...")
    logger.info("-" * 60)

    results = {}
    for name, extractor in extractors.items():
        try:
            features = extractor.extract_features(audio, sr=sr)
            results[name] = features
            logger.info(f"{name:15} | Shape: {features.shape:20} | Range: [{features.min():.4f}, {features.max():.4f}]")
        except Exception as e:
            logger.warning(f"{name}: {str(e)[:50]}")

    logger.info("-" * 60)

    # Compare feature statistics
    if results:
        logger.info("Feature statistics:")
        for name, features in results.items():
            logger.info(
                f"{name:15} | Mean: {features.mean():.4f}, Std: {features.std():.4f}, Dim: {features.shape[-1]}"
            )


def example_preprocessing_plus_foundation():
    """Example: Combine traditional preprocessing with foundation models."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Hybrid Preprocessing Approach")
    logger.info("=" * 60)

    audio, sr = generate_synthetic_audio()

    # Traditional preprocessing
    logger.info("Traditional preprocessing...")
    processor = AudioProcessor()
    mfcc = processor.extract_mfcc(audio, sr=sr)
    mel_spec = processor.extract_mel_spectrogram(audio, sr=sr)

    logger.info(f"MFCC shape: {mfcc.shape}")
    logger.info(f"Mel-spectrogram shape: {mel_spec.shape}")

    # Foundation model extraction
    logger.info("Foundation model feature extraction...")
    try:
        wav2vec2 = Wav2Vec2FeatureExtractor()
        foundation_features = wav2vec2.extract_features(audio, sr=sr)
        logger.info(f"Foundation features shape: {foundation_features.shape}")

        # Stack features for ensemble input
        # Traditional features: (39, T) -> expand to (39, 256)
        # Foundation features: (1, T, 768) -> time-average to (768,)
        logger.info("Combined approach enables:")
        logger.info("  - Low-level spectral patterns from MFCC/spectrogram")
        logger.info("  - High-level semantic patterns from foundation models")
        logger.info("  - Improved robustness to different deepfake types")

    except ImportError as e:
        logger.warning(f"Foundation models not available: {e}")


def example_transfer_learning_preparation():
    """Example: Prepare features for transfer learning."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 7: Transfer Learning Feature Preparation")
    logger.info("=" * 60)

    # Simulate batch of audio files
    n_files = 5
    audio_files = [generate_synthetic_audio()[0] for _ in range(n_files)]

    logger.info(f"Processing {n_files} audio files...")

    try:
        extractor = Wav2Vec2FeatureExtractor()

        logger.info("Extracting features for transfer learning...")
        features_batch = []

        for i, audio in enumerate(audio_files):
            features = extractor.extract_features(audio, sr=16000)
            features_batch.append(features)

            logger.info(f"File {i+1}: features shape {features.shape}")

        # Stack all features
        features_batch = np.vstack(features_batch)
        logger.info(f"Batch shape: {features_batch.shape}")

        logger.info("Features ready for:")
        logger.info("  - Fine-tuning downstream deepfake detection models")
        logger.info("  - Zero-shot classification with pre-trained models")
        logger.info("  - Cross-dataset evaluation")

    except ImportError as e:
        logger.warning(f"Transfer learning features not available: {e}")


if __name__ == "__main__":
    # Run examples
    example_wav2vec2_extraction()
    print("\n")

    example_hubert_extraction()
    print("\n")

    example_whisper_extraction()
    print("\n")

    example_foundation_model_ensemble()
    print("\n")

    example_feature_comparison()
    print("\n")

    example_preprocessing_plus_foundation()
    print("\n")

    example_transfer_learning_preparation()

    logger.info("All foundation model examples completed!")
