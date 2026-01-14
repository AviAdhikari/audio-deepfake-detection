"""Example: Explainable AI visualization with SHAP and Grad-CAM."""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xai.interpretability import (
    GradCAM,
    SHAPExplainer,
    IntegratedGradients,
    XAIVisualizer,
    SaliencyMap,
)
from src.models.hybrid_model import HybridDeepfakeDetector
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_synthetic_data(n_samples: int = 5):
    """Generate synthetic audio features."""
    logger.info(f"Generating {n_samples} synthetic samples...")

    X = np.random.randn(n_samples, 2, 39, 256).astype(np.float32)
    # Normalize
    X = (X - X.mean(axis=(1, 2, 3), keepdims=True)) / (
        X.std(axis=(1, 2, 3), keepdims=True) + 1e-7
    )

    return X


def example_gradcam_visualization():
    """Example: Visualize predictions with Grad-CAM."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Grad-CAM Visualization")
    logger.info("=" * 60)

    # Create and compile model
    logger.info("Building model...")
    model = HybridDeepfakeDetector(input_shape=(2, 39, 256))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Generate test data
    X_test = generate_synthetic_data(n_samples=1)

    # Initialize Grad-CAM
    try:
        logger.info("Initializing Grad-CAM for 'conv2d' layer...")
        gradcam = GradCAM(model, layer_name="conv2d")

        # Compute heatmap
        logger.info("Computing heatmap...")
        heatmap = gradcam.compute_heatmap(X_test)

        logger.info(f"Heatmap shape: {heatmap.shape}")
        logger.info(f"Heatmap range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
        logger.info("Grad-CAM shows which CNN filters are most active in deepfake detection")

    except Exception as e:
        logger.warning(f"Grad-CAM computation: {e}")


def example_integrated_gradients():
    """Example: Integrated Gradients attribution."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Integrated Gradients Attribution")
    logger.info("=" * 60)

    # Create and compile model
    logger.info("Building model...")
    model = HybridDeepfakeDetector(input_shape=(2, 39, 256))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Generate test data
    X_test = generate_synthetic_data(n_samples=1)

    # Initialize Integrated Gradients
    logger.info("Initializing Integrated Gradients...")
    ig = IntegratedGradients(model)

    try:
        # Compute attribution
        logger.info("Computing integrated gradients...")
        attributions = ig.integrated_gradients(X_test)

        logger.info(f"Attribution shape: {attributions.shape}")
        logger.info(f"Attribution range: [{attributions.min():.4f}, {attributions.max():.4f}]")

        # Get summary
        summary = ig.attribution_summary(X_test)
        logger.info(f"Top important channel: {summary['top_channels']}")
        logger.info(f"Max importance: {summary['max_importance']:.4f}")
        logger.info(
            "Integrated Gradients shows contribution of each input feature to prediction"
        )

    except Exception as e:
        logger.warning(f"Integrated Gradients: {e}")


def example_saliency_map():
    """Example: Saliency map visualization."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Saliency Map Visualization")
    logger.info("=" * 60)

    # Create and compile model
    logger.info("Building model...")
    model = HybridDeepfakeDetector(input_shape=(2, 39, 256))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Generate test data
    X_test = generate_synthetic_data(n_samples=1)

    # Initialize saliency map
    logger.info("Initializing saliency map generator...")
    saliency = SaliencyMap(model)

    try:
        # Compute saliency
        logger.info("Computing saliency map...")
        saliency_map = saliency.compute_saliency(X_test)

        logger.info(f"Saliency shape: {saliency_map.shape}")
        logger.info(f"Saliency range: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
        logger.info("Saliency shows pixel-level importance for the prediction")

    except Exception as e:
        logger.warning(f"Saliency map computation: {e}")


def example_shap_explanation():
    """Example: SHAP explanations."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: SHAP Explanations")
    logger.info("=" * 60)

    # Create and compile model
    logger.info("Building model...")
    model = HybridDeepfakeDetector(input_shape=(2, 39, 256))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Generate test data
    X_test = generate_synthetic_data(n_samples=1)
    X_background = generate_synthetic_data(n_samples=10)

    # Initialize SHAP
    logger.info("Initializing SHAP explainer...")
    shap_explainer = SHAPExplainer(model, background_data=X_background)

    try:
        # Generate explanation
        logger.info("Computing SHAP values...")
        explanation = shap_explainer.explain_prediction(X_test)

        logger.info(f"Prediction: {explanation['prediction']:.4f}")
        logger.info("SHAP explanation generated")
        logger.info("SHAP values show each feature's contribution to pushing prediction away from baseline")

        # Save explanation
        logger.info("Saving explanation to file...")
        output_path = Path(__file__).parent.parent / "shap_explanation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(explanation, f, indent=2, default=str)

        logger.info(f"Explanation saved to {output_path}")

    except ImportError:
        logger.warning("SHAP not installed. Install with: pip install shap")
    except Exception as e:
        logger.warning(f"SHAP explanation: {e}")


def example_unified_xai():
    """Example: Unified XAI with multiple explanation methods."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Unified XAI Visualization")
    logger.info("=" * 60)

    # Create and compile model
    logger.info("Building model...")
    model = HybridDeepfakeDetector(input_shape=(2, 39, 256))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Generate test data
    X_test = generate_synthetic_data(n_samples=1)
    original_spec = np.random.randn(128, 256)  # Simulated mel-spectrogram

    # Initialize unified XAI
    logger.info("Initializing unified XAI visualizer...")
    xai = XAIVisualizer(model, layer_name="conv2d")

    try:
        # Generate comprehensive explanation
        logger.info("Generating comprehensive explanation...")
        explanation = xai.explain_prediction(X_test, original_spectrogram=original_spec)

        logger.info(f"Prediction: {explanation['prediction']['probability']:.4f}")
        logger.info(f"Is deepfake: {explanation['prediction']['is_deepfake']}")
        logger.info(f"Confidence: {explanation['prediction']['confidence']:.4f}")
        logger.info(f"Grad-CAM heatmap shape: {len(explanation['gradcam_heatmap'])}")
        logger.info(f"IG top channel: {explanation['integrated_gradients']['top_channels']}")

        # Save explanation
        logger.info("Saving comprehensive explanation...")
        output_path = Path(__file__).parent.parent / "xai_explanation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        xai.save_explanation(explanation, str(output_path))
        logger.info(f"Explanation saved to {output_path}")

        logger.info("Unified XAI combines:")
        logger.info("  - Grad-CAM: Shows which CNN filters activate")
        logger.info("  - Integrated Gradients: Shows feature attribution")
        logger.info("  - Prediction confidence: Shows model certainty")

    except Exception as e:
        logger.warning(f"Unified XAI: {e}")


def example_batch_explanation():
    """Example: Explain predictions for multiple samples."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Batch Explanation")
    logger.info("=" * 60)

    # Create and compile model
    logger.info("Building model...")
    model = HybridDeepfakeDetector(input_shape=(2, 39, 256))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Generate test data
    X_test = generate_synthetic_data(n_samples=5)

    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X_test, verbose=0)

    logger.info("Batch predictions:")
    logger.info("-" * 60)

    for i, pred in enumerate(predictions):
        prob = float(pred[0])
        is_deepfake = "DEEPFAKE" if prob >= 0.5 else "GENUINE"
        confidence = max(prob, 1 - prob)

        logger.info(f"Sample {i+1}: {is_deepfake:10} | Confidence: {confidence:.4f}")

    logger.info("-" * 60)
    logger.info("For detailed explanations, choose specific samples of interest")


def example_false_positive_analysis():
    """Example: Analyze false positive predictions."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 7: False Positive Analysis")
    logger.info("=" * 60)

    # Create and compile model
    logger.info("Building model...")
    model = HybridDeepfakeDetector(input_shape=(2, 39, 256))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Create test set with known labels
    X_genuine = generate_synthetic_data(n_samples=10)
    y_genuine = np.zeros(10)  # All genuine

    # Make predictions
    logger.info("Predicting on genuine samples...")
    predictions = model.predict(X_genuine, verbose=0)

    # Find false positives
    false_positives = predictions[:, 0] >= 0.5
    n_false_pos = np.sum(false_positives)

    logger.info(f"False positives: {n_false_pos}/{len(X_genuine)}")

    if n_false_pos > 0:
        logger.info("Analyzing first false positive...")

        # Get first false positive
        fp_idx = np.where(false_positives)[0][0]
        X_fp = X_genuine[fp_idx : fp_idx + 1]

        # Explain it
        xai = XAIVisualizer(model, layer_name="conv2d")

        try:
            explanation = xai.explain_prediction(X_fp)
            logger.info(f"FP sample prediction: {explanation['prediction']['probability']:.4f}")
            logger.info("Use XAI methods above to understand why this sample was misclassified")

        except Exception as e:
            logger.warning(f"FP analysis: {e}")
    else:
        logger.info("No false positives found in test set")


if __name__ == "__main__":
    # Run examples
    example_gradcam_visualization()
    print("\n")

    example_integrated_gradients()
    print("\n")

    example_saliency_map()
    print("\n")

    example_shap_explanation()
    print("\n")

    example_unified_xai()
    print("\n")

    example_batch_explanation()
    print("\n")

    example_false_positive_analysis()

    logger.info("All XAI examples completed!")
