"""
Example script for audio preprocessing.

Usage:
    python examples/preprocessing_example.py --audio audio.wav
"""

import argparse
import logging
import numpy as np
from pathlib import Path

from src.preprocessing import AudioProcessor
from src.utils import setup_logging


def main(audio_path: str):
    """Demonstrate audio preprocessing."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("Audio Preprocessing Example")
    logger.info("=" * 50)

    # Initialize processor
    processor = AudioProcessor(
        target_sr=16000,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        n_mfcc=13,
    )

    # Check if audio file exists
    if not Path(audio_path).exists():
        logger.warning(
            f"Audio file not found: {audio_path}"
        )
        logger.info("Creating synthetic audio for demonstration...")
        # Create synthetic audio
        sr = 16000
        duration = 5  # seconds
        t = np.arange(0, duration, 1 / sr)
        # Generate a mix of sine waves
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz
        y += 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz
        audio_path = "synthetic_audio.wav"
        import scipy.io.wavfile as wavfile

        wavfile.write(audio_path, sr, (y * 32767).astype(np.int16))
        logger.info(f"Synthetic audio saved to {audio_path}")

    # Process single audio file
    logger.info(f"\nProcessing audio: {audio_path}")
    features = processor.process_audio(audio_path)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"  Channels: {features.shape[0]}")
    logger.info(f"  Feature dimension: {features.shape[1]}")
    logger.info(f"  Time steps: {features.shape[2]}")

    # Extract individual features
    logger.info("\nExtracting individual features...")
    y, sr = processor.load_audio(audio_path)

    logger.info(f"Audio loaded:")
    logger.info(f"  Duration: {len(y) / sr:.2f} seconds")
    logger.info(f"  Sampling rate: {sr} Hz")

    # MFCC
    mfcc = processor.extract_mfcc(y, sr)
    logger.info(f"MFCC shape: {mfcc.shape}")

    # Mel spectrogram
    mel_spec = processor.extract_mel_spectrogram(y, sr)
    logger.info(f"Mel spectrogram shape: {mel_spec.shape}")

    # STFT
    stft = processor.extract_stft(y)
    logger.info(f"STFT magnitude shape: {stft.shape}")

    # Chroma
    chroma = processor.extract_chroma(y, sr)
    logger.info(f"Chroma features shape: {chroma.shape}")

    # Spectral centroid
    spectral_centroid = processor.extract_spectral_centroid(y, sr)
    logger.info(f"Spectral centroid shape: {spectral_centroid.shape}")

    logger.info("\n" + "=" * 50)
    logger.info("Preprocessing complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio preprocessing example")
    parser.add_argument(
        "--audio",
        type=str,
        default="sample_audio.wav",
        help="Path to audio file",
    )
    args = parser.parse_args()

    main(args.audio)
