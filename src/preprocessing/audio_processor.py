"""Audio preprocessing module for feature extraction."""

import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Tuple, Optional
from scipy import signal

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Preprocesses audio files for deepfake detection."""

    def __init__(
        self,
        target_sr: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mfcc: int = 13,
    ):
        """
        Initialize audio processor.

        Args:
            target_sr: Target sampling rate (Hz)
            n_mels: Number of Mel bands
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mfcc: Number of MFCC coefficients
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def load_audio(
        self, audio_path: str, duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with resampling.

        Args:
            audio_path: Path to audio file
            duration: Maximum duration in seconds (None for full audio)

        Returns:
            Tuple of (audio_data, sampling_rate)
        """
        try:
            y, sr = librosa.load(
                audio_path, sr=self.target_sr, duration=duration, mono=True
            )
            logger.info(f"Loaded audio from {audio_path}: shape={y.shape}, sr={sr}")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio from {audio_path}: {e}")
            raise

    def extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract Mel-frequency cepstral coefficients.

        Args:
            y: Audio time series
            sr: Sampling rate

        Returns:
            MFCC features (n_mfcc, time_steps)
        """
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        # Add delta (first derivative) and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack features (3, n_mfcc, time_steps)
        stacked = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        logger.debug(f"MFCC features shape: {stacked.shape}")
        return stacked

    def extract_mel_spectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract log-mel spectrogram.

        Args:
            y: Audio time series
            sr: Sampling rate

        Returns:
            Log-mel spectrogram (n_mels, time_steps)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        logger.debug(f"Log-mel spectrogram shape: {log_mel_spec.shape}")
        return log_mel_spec

    def extract_stft(self, y: np.ndarray) -> np.ndarray:
        """
        Extract short-time Fourier transform magnitude.

        Args:
            y: Audio time series

        Returns:
            STFT magnitude (n_fft//2 + 1, time_steps)
        """
        stft_matrix = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft_matrix)
        # Convert to log scale
        log_stft = librosa.power_to_db(magnitude, ref=np.max)
        logger.debug(f"Log STFT shape: {log_stft.shape}")
        return log_stft

    def extract_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract chroma features.

        Args:
            y: Audio time series
            sr: Sampling rate

        Returns:
            Chroma features (12, time_steps)
        """
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        logger.debug(f"Chroma features shape: {chroma.shape}")
        return chroma

    def extract_spectral_centroid(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract spectral centroid features.

        Args:
            y: Audio time series
            sr: Sampling rate

        Returns:
            Spectral centroid (1, time_steps)
        """
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        logger.debug(f"Spectral centroid shape: {spectral_centroid.shape}")
        return spectral_centroid

    def pad_or_truncate(
        self, features: np.ndarray, target_length: int
    ) -> np.ndarray:
        """
        Pad or truncate features to target length.

        Args:
            features: Input features (channels, time_steps)
            target_length: Target length in time steps

        Returns:
            Padded/truncated features
        """
        current_length = features.shape[-1]

        if current_length < target_length:
            # Pad with zeros
            pad_width = ((0, 0),) * (features.ndim - 1) + (
                (0, target_length - current_length),
            )
            padded = np.pad(features, pad_width, mode="constant", constant_values=0)
            return padded
        else:
            # Truncate
            return features[..., :target_length]

    def process_audio(
        self, audio_path: str, duration: Optional[float] = None, target_length: int = 256
    ) -> np.ndarray:
        """
        Complete audio processing pipeline.

        Args:
            audio_path: Path to audio file
            duration: Maximum duration in seconds
            target_length: Target time steps (256 ≈ 8.5 seconds at 16kHz)

        Returns:
            Multi-channel features (channels, height, width) where:
            - channels=2 (MFCC+delta+delta2, Log-mel spectrogram)
            - height=features dimension
            - width=time steps
        """
        # Load audio
        y, sr = self.load_audio(audio_path, duration)

        # Extract features
        mfcc_features = self.extract_mfcc(y, sr)  # (39, T)
        mel_features = self.extract_mel_spectrogram(y, sr)  # (128, T)

        # Normalize features
        mfcc_features = self._normalize_features(mfcc_features)
        mel_features = self._normalize_features(mel_features)

        # Pad or truncate to target length
        mfcc_features = self.pad_or_truncate(mfcc_features, target_length)
        mel_features = self.pad_or_truncate(mel_features, target_length)

        # Stack into multi-channel input (2, feature_dim, time_steps)
        # Channel 0: MFCC features, Channel 1: Mel spectrogram
        multi_channel = np.stack([mfcc_features, mel_features], axis=0)

        logger.info(
            f"Processed audio shape: {multi_channel.shape} "
            f"(channels={multi_channel.shape[0]}, "
            f"height={multi_channel.shape[1]}, "
            f"width={multi_channel.shape[2]})"
        )

        return multi_channel

    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        """
        Normalize features using mean and standard deviation.

        Args:
            features: Input features

        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=-1, keepdims=True)
        std = np.std(features, axis=-1, keepdims=True)
        # Avoid division by zero
        std[std == 0] = 1.0
        return (features - mean) / std

    def batch_process(
        self,
        audio_paths: list,
        duration: Optional[float] = None,
        target_length: int = 256,
    ) -> Tuple[np.ndarray, list]:
        """
        Process multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            duration: Maximum duration in seconds
            target_length: Target time steps

        Returns:
            Tuple of (batch_features, valid_paths)
        """
        batch_features = []
        valid_paths = []

        for audio_path in audio_paths:
            try:
                features = self.process_audio(audio_path, duration, target_length)
                batch_features.append(features)
                valid_paths.append(audio_path)
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                continue

        if not batch_features:
            raise ValueError("No audio files were successfully processed")

        # Stack into batch (batch_size, channels, height, width)
        batch = np.stack(batch_features, axis=0)
        logger.info(f"Batch processing complete: shape={batch.shape}")

        return batch, valid_paths
