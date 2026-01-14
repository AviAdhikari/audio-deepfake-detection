"""Audio deepfake detector for inference."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from datetime import datetime

from src.preprocessing import AudioProcessor

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """
    Production-ready deepfake detector for audio files.
    
    Performs preprocessing, inference, and threshold-based classification.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        target_sr: int = 16000,
        target_length: int = 256,
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to trained Keras model
            threshold: Classification threshold (probability >= threshold = deepfake)
            target_sr: Target sampling rate
            target_length: Target time steps for audio preprocessing
        """
        self.threshold = threshold
        self.target_sr = target_sr
        self.target_length = target_length

        # Load model
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Initialize audio processor
        self.audio_processor = AudioProcessor(target_sr=target_sr)
        logger.info("Audio processor initialized")

    def detect_single(
        self, audio_path: str, return_confidence: bool = True
    ) -> Dict:
        """
        Detect if a single audio file is a deepfake.

        Args:
            audio_path: Path to audio file
            return_confidence: Whether to return confidence scores

        Returns:
            Dictionary containing:
                - is_deepfake: Boolean classification
                - probability: Probability of being deepfake
                - confidence: Confidence in prediction
                - timestamp: Inference timestamp
                - threshold_used: Classification threshold
        """
        try:
            # Preprocess audio
            features = self.audio_processor.process_audio(
                audio_path, target_length=self.target_length
            )

            # Add batch dimension
            features_batch = np.expand_dims(features, axis=0)

            # Run inference
            probability = self.model.predict(features_batch, verbose=0)[0, 0]

            # Classify
            is_deepfake = probability >= self.threshold

            # Calculate confidence
            if is_deepfake:
                confidence = probability
            else:
                confidence = 1 - probability

            result = {
                "audio_file": str(audio_path),
                "is_deepfake": bool(is_deepfake),
                "probability": float(probability),
                "confidence": float(confidence),
                "threshold_used": self.threshold,
                "timestamp": datetime.now().isoformat(),
                "success": True,
            }

            logger.info(
                f"Detection result for {audio_path}: "
                f"deepfake={is_deepfake}, probability={probability:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error detecting {audio_path}: {e}")
            return {
                "audio_file": str(audio_path),
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def detect_batch(
        self,
        audio_paths: List[str],
        batch_size: int = 32,
        return_sorted: bool = True,
    ) -> Dict:
        """
        Detect deepfakes in multiple audio files.

        Args:
            audio_paths: List of paths to audio files
            batch_size: Batch size for inference
            return_sorted: Sort results by probability (descending)

        Returns:
            Dictionary containing:
                - results: List of detection results
                - summary: Statistics about the batch
        """
        results = []
        successful = 0
        failed = 0

        logger.info(f"Processing batch of {len(audio_paths)} audio files")

        for i, audio_path in enumerate(audio_paths, 1):
            result = self.detect_single(audio_path)
            results.append(result)

            if result["success"]:
                successful += 1
            else:
                failed += 1

            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(audio_paths)} files")

        # Sort by probability if requested
        if return_sorted:
            results = sorted(
                [r for r in results if r["success"]],
                key=lambda x: x["probability"],
                reverse=True,
            ) + [r for r in results if not r["success"]]

        # Calculate summary statistics
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            probabilities = [r["probability"] for r in successful_results]
            deepfakes = [r["is_deepfake"] for r in successful_results]

            summary = {
                "total_processed": len(audio_paths),
                "successful": successful,
                "failed": failed,
                "deepfakes_detected": sum(deepfakes),
                "legit_detected": len(deepfakes) - sum(deepfakes),
                "average_probability": float(np.mean(probabilities)),
                "min_probability": float(np.min(probabilities)),
                "max_probability": float(np.max(probabilities)),
                "threshold_used": self.threshold,
            }
        else:
            summary = {
                "total_processed": len(audio_paths),
                "successful": 0,
                "failed": failed,
                "deepfakes_detected": 0,
                "legit_detected": 0,
                "error": "No files processed successfully",
            }

        batch_result = {
            "results": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"Batch processing complete: "
            f"deepfakes={summary['deepfakes_detected']}, "
            f"legit={summary['legit_detected']}, "
            f"failed={failed}"
        )

        return batch_result

    def set_threshold(self, threshold: float):
        """
        Update classification threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model.name,
            "trainable_params": self.model.count_params(),
            "layers": len(self.model.layers),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
        }

    def export_results(self, batch_results: Dict, export_path: str):
        """
        Export batch results to JSON file.

        Args:
            batch_results: Results from detect_batch
            export_path: Path to save results JSON
        """
        import json

        try:
            with open(export_path, "w") as f:
                json.dump(batch_results, f, indent=2)
            logger.info(f"Results exported to {export_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise

    def get_prediction_distribution(self, probabilities: np.ndarray) -> Dict:
        """
        Get distribution statistics of predictions.

        Args:
            probabilities: Array of prediction probabilities

        Returns:
            Distribution statistics
        """
        return {
            "mean": float(np.mean(probabilities)),
            "median": float(np.median(probabilities)),
            "std": float(np.std(probabilities)),
            "min": float(np.min(probabilities)),
            "max": float(np.max(probabilities)),
            "q25": float(np.percentile(probabilities, 25)),
            "q75": float(np.percentile(probabilities, 75)),
        }
