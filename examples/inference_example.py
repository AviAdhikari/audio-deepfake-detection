"""
Example script for running inference with the deepfake detector.

Usage:
    python examples/inference_example.py --model models/deepfake_detector.keras --audio audio.wav
"""

import argparse
import logging
import json
from pathlib import Path

from src.inference import DeepfakeDetector
from src.utils import setup_logging


def main(model_path: str, audio_paths: list, threshold: float = 0.5):
    """Run inference on audio files."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("Audio Deepfake Detection - Inference")
    logger.info("=" * 50)

    # Validate model path
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return

    # Load detector
    logger.info(f"Loading model from {model_path}...")
    detector = DeepfakeDetector(
        model_path=model_path, threshold=threshold
    )

    # Get model info
    model_info = detector.get_model_info()
    logger.info(f"Model info: {model_info}")

    # Single file detection example
    if len(audio_paths) == 1:
        logger.info(f"\nDetecting deepfake in: {audio_paths[0]}")
        result = detector.detect_single(audio_paths[0])

        if result["success"]:
            logger.info(f"Result: {json.dumps(result, indent=2)}")
        else:
            logger.error(f"Detection failed: {result['error']}")

    # Batch detection
    else:
        logger.info(f"\nRunning batch detection on {len(audio_paths)} files...")
        batch_results = detector.detect_batch(audio_paths)

        # Print summary
        summary = batch_results["summary"]
        logger.info(f"\nBatch Summary:")
        logger.info(f"  Total processed: {summary['total_processed']}")
        logger.info(f"  Successful: {summary['successful']}")
        logger.info(f"  Failed: {summary['failed']}")
        logger.info(f"  Deepfakes detected: {summary['deepfakes_detected']}")
        logger.info(f"  Legitimate detected: {summary['legit_detected']}")
        logger.info(f"  Average probability: {summary['average_probability']:.4f}")

        # Save results
        output_path = "detection_results.json"
        detector.export_results(batch_results, output_path)
        logger.info(f"\nResults saved to {output_path}")

        # Print individual results
        logger.info("\nIndividual Results:")
        for result in batch_results["results"][:10]:  # Print first 10
            if result["success"]:
                logger.info(
                    f"  {Path(result['audio_file']).name}: "
                    f"deepfake={result['is_deepfake']}, "
                    f"prob={result['probability']:.4f}"
                )

    logger.info("\n" + "=" * 50)
    logger.info("Inference complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake detection inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to audio file(s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    args = parser.parse_args()

    main(args.model, args.audio, args.threshold)
