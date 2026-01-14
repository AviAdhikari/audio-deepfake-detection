"""
Training script for ASVspoof2019 and WaveFake benchmark datasets.

This script trains the deepfake detection models on standard benchmarks
for publication-quality evaluation.

Usage:
    python examples/train_on_asvspoof_wavefake.py
"""

import os
import json
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from src.models import HybridDeepfakeDetector, TransformerDeepfakeDetector
from src.models.foundation_models import Wav2Vec2FeatureExtractor
from src.preprocessing import AudioProcessor
from src.training import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ASVspoofDataLoader:
    """Load ASVspoof2019 LA (Logical Access) dataset."""

    def __init__(self, data_dir: str = "data/ASVspoof2019"):
        """
        Initialize ASVspoof data loader.

        Args:
            data_dir: Path to ASVspoof2019 directory
        """
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor(target_sr=16000)

    def load_dataset(
        self, subset: str = "LA", split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ASVspoof2019 dataset.

        Args:
            subset: "LA" (logical access) or "PA" (physical access)
            split: "train", "dev", or "eval"

        Returns:
            (X, y) where X has shape (N, 2, 39, 256) and y has shape (N,)
        """
        # Expected path: data/ASVspoof2019/ASVspoof2019_LA_train/
        subset_dir = self.data_dir / f"ASVspoof2019_{subset}_{split}"

        if not subset_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {subset_dir}\n"
                f"Download from: https://datashare.ed.ac.uk/handle/10283/3336"
            )

        # Load protocol file
        protocol_file = subset_dir / "protocol.txt"
        audio_dir = subset_dir / "flac"

        features_list = []
        labels_list = []

        logger.info(f"Loading {subset} {split} split...")

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                # Format: speaker_id audio_file - - label
                audio_file = parts[1]
                label = parts[-1]  # "bonafide" or "spoof"

                # Convert label to binary: 0=genuine, 1=spoofed
                y = 0 if label == "bonafide" else 1

                audio_path = audio_dir / f"{audio_file}.flac"

                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    continue

                try:
                    # Load and process audio
                    audio, sr = librosa.load(str(audio_path), sr=16000)

                    # Extract MFCC features
                    mfcc = librosa.feature.mfcc(
                        y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
                    )

                    # Extract delta (velocity)
                    delta = librosa.feature.delta(mfcc)

                    # Stack features: (2, 13, T) -> (2, 13, 256)
                    # Pad or truncate to 256 frames
                    if mfcc.shape[1] < 256:
                        mfcc = np.pad(mfcc, ((0, 0), (0, 256 - mfcc.shape[1])))
                        delta = np.pad(delta, ((0, 0), (0, 256 - delta.shape[1])))
                    else:
                        mfcc = mfcc[:, :256]
                        delta = delta[:, :256]

                    features = np.stack([mfcc, delta])  # Shape: (2, 13, 256)
                    features_list.append(features)
                    labels_list.append(y)

                except Exception as e:
                    logger.warning(f"Error processing {audio_path}: {e}")
                    continue

        if not features_list:
            raise RuntimeError(f"No audio files loaded from {subset_dir}")

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"Loaded {len(X)} samples from {subset} {split} split")
        logger.info(f"Label distribution: {np.bincount(y)}")

        return X, y


class WaveFakeDataLoader:
    """Load WaveFake dataset."""

    def __init__(self, data_dir: str = "data/WaveFake"):
        """
        Initialize WaveFake data loader.

        Args:
            data_dir: Path to WaveFake directory
        """
        self.data_dir = Path(data_dir)
        self.audio_processor = AudioProcessor(target_sr=16000)

    def load_dataset(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load WaveFake dataset.

        Args:
            split: "train", "val", or "test"

        Returns:
            (X, y) where X has shape (N, 2, 39, 256) and y has shape (N,)
        """
        split_dir = self.data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {split_dir}\n"
                f"Download from: https://zenodo.org/record/3629246"
            )

        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        features_list = []
        labels_list = []

        logger.info(f"Loading WaveFake {split} split...")

        # Load real audio (label 0)
        for audio_file in real_dir.glob("*.wav"):
            try:
                audio, sr = librosa.load(str(audio_file), sr=16000)
                mfcc = librosa.feature.mfcc(
                    y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
                )
                delta = librosa.feature.delta(mfcc)

                if mfcc.shape[1] < 256:
                    mfcc = np.pad(mfcc, ((0, 0), (0, 256 - mfcc.shape[1])))
                    delta = np.pad(delta, ((0, 0), (0, 256 - delta.shape[1])))
                else:
                    mfcc = mfcc[:, :256]
                    delta = delta[:, :256]

                features = np.stack([mfcc, delta])
                features_list.append(features)
                labels_list.append(0)  # Real

            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")

        # Load fake audio (label 1)
        for audio_file in fake_dir.glob("*.wav"):
            try:
                audio, sr = librosa.load(str(audio_file), sr=16000)
                mfcc = librosa.feature.mfcc(
                    y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
                )
                delta = librosa.feature.delta(mfcc)

                if mfcc.shape[1] < 256:
                    mfcc = np.pad(mfcc, ((0, 0), (0, 256 - mfcc.shape[1])))
                    delta = np.pad(delta, ((0, 0), (0, 256 - delta.shape[1])))
                else:
                    mfcc = mfcc[:, :256]
                    delta = delta[:, :256]

                features = np.stack([mfcc, delta])
                features_list.append(features)
                labels_list.append(1)  # Fake

            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")

        if not features_list:
            raise RuntimeError(f"No audio files loaded from {split_dir}")

        X = np.array(features_list)
        y = np.array(labels_list)

        logger.info(f"Loaded {len(X)} samples from WaveFake {split} split")
        logger.info(f"Label distribution: {np.bincount(y)}")

        return X, y


def train_models_on_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str = "benchmark",
) -> Dict:
    """
    Train multiple models on the dataset.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of dataset for saving

    Returns:
        Dictionary with results
    """
    results = {}

    # Split training data for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    logger.info(f"Training set: {len(X_tr)}, Validation set: {len(X_val)}")

    # Model 1: HybridDeepfakeDetector
    logger.info("Training HybridDeepfakeDetector...")
    model1 = HybridDeepfakeDetector(input_shape=(2, 13, 256))
    trainer1 = Trainer(model1, config={"epochs": 30, "batch_size": 32})
    history1 = trainer1.train(X_tr, y_tr, X_val, y_val)

    # Evaluate
    pred1 = model1.predict(X_test)
    pred1_binary = (pred1 > 0.5).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    results["HybridDeepfakeDetector"] = {
        "accuracy": float(accuracy_score(y_test, pred1_binary)),
        "precision": float(precision_score(y_test, pred1_binary)),
        "recall": float(recall_score(y_test, pred1_binary)),
        "f1_score": float(f1_score(y_test, pred1_binary)),
        "roc_auc": float(roc_auc_score(y_test, pred1)),
        "y_pred": pred1_binary.tolist(),
        "y_pred_proba": pred1.flatten().tolist(),
        "history": {
            "loss": history1["loss"],
            "val_loss": history1.get("val_loss", []),
            "accuracy": history1.get("accuracy", []),
            "val_accuracy": history1.get("val_accuracy", []),
        },
    }

    logger.info(
        f"HybridDeepfakeDetector - Accuracy: {results['HybridDeepfakeDetector']['accuracy']:.4f}"
    )

    # Model 2: TransformerDeepfakeDetector
    logger.info("Training TransformerDeepfakeDetector...")
    model2 = TransformerDeepfakeDetector(input_shape=(2, 13, 256))
    trainer2 = Trainer(model2, config={"epochs": 30, "batch_size": 32})
    history2 = trainer2.train(X_tr, y_tr, X_val, y_val)

    pred2 = model2.predict(X_test)
    pred2_binary = (pred2 > 0.5).astype(int)

    results["TransformerDeepfakeDetector"] = {
        "accuracy": float(accuracy_score(y_test, pred2_binary)),
        "precision": float(precision_score(y_test, pred2_binary)),
        "recall": float(recall_score(y_test, pred2_binary)),
        "f1_score": float(f1_score(y_test, pred2_binary)),
        "roc_auc": float(roc_auc_score(y_test, pred2)),
        "y_pred": pred2_binary.tolist(),
        "y_pred_proba": pred2.flatten().tolist(),
        "history": {
            "loss": history2["loss"],
            "val_loss": history2.get("val_loss", []),
            "accuracy": history2.get("accuracy", []),
            "val_accuracy": history2.get("val_accuracy", []),
        },
    }

    logger.info(
        f"TransformerDeepfakeDetector - Accuracy: {results['TransformerDeepfakeDetector']['accuracy']:.4f}"
    )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"{dataset_name.lower()}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Save models
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model1.save(str(models_dir / f"HybridDeepfakeDetector_{dataset_name}.keras"))
    model2.save(str(models_dir / f"TransformerDeepfakeDetector_{dataset_name}.keras"))

    logger.info("Models saved to models/")

    return results


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Training on Benchmark Datasets")
    logger.info("=" * 60)

    # Try ASVspoof
    try:
        logger.info("\n" + "=" * 60)
        logger.info("ASVspoof2019 Dataset")
        logger.info("=" * 60)

        loader_asvspoof = ASVspoofDataLoader("data/ASVspoof2019")
        X_train, y_train = loader_asvspoof.load_dataset(subset="LA", split="train")
        X_test, y_test = loader_asvspoof.load_dataset(subset="LA", split="eval")

        results_asvspoof = train_models_on_dataset(
            X_train, y_train, X_test, y_test, "ASVspoof2019"
        )

        logger.info("\nASVspoof2019 Results:")
        for model_name, metrics in results_asvspoof.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    except FileNotFoundError as e:
        logger.error(f"ASVspoof dataset not available: {e}")
        logger.info("Download from: https://datashare.ed.ac.uk/handle/10283/3336")

    # Try WaveFake
    try:
        logger.info("\n" + "=" * 60)
        logger.info("WaveFake Dataset")
        logger.info("=" * 60)

        loader_wavefake = WaveFakeDataLoader("data/WaveFake")
        X_train, y_train = loader_wavefake.load_dataset(split="train")
        X_test, y_test = loader_wavefake.load_dataset(split="test")

        results_wavefake = train_models_on_dataset(
            X_train, y_train, X_test, y_test, "WaveFake"
        )

        logger.info("\nWaveFake Results:")
        for model_name, metrics in results_wavefake.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    except FileNotFoundError as e:
        logger.error(f"WaveFake dataset not available: {e}")
        logger.info("Download from: https://zenodo.org/record/3629246")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
