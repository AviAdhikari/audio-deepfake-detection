"""Explainable AI (XAI) module for deepfake detection visualization."""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visualization.
    
    Shows which regions of the spectrogram are important for the deepfake
    classification decision.
    """

    def __init__(self, model, layer_name: str):
        """
        Initialize Grad-CAM visualizer.

        Args:
            model: Keras model
            layer_name: Name of layer to visualize
        """
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(layer_name).output,
                model.output,
            ],
        )
        logger.info(f"Initialized Grad-CAM for layer: {layer_name}")

    def compute_heatmap(self, input_data: np.ndarray, class_idx: int = 1) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.

        Args:
            input_data: Input features (1, channels, height, width)
            class_idx: Class index (1 for deepfake, 0 for real)

        Returns:
            Heatmap showing important regions
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(input_data)
            loss = predictions[:, 0]

        # Get gradients
        grads = tape.gradient(loss, conv_outputs)

        # Compute weights as mean across spatial dimensions
        weights = tf.reduce_mean(grads, axis=(1, 2))

        # Compute weighted sum of feature maps
        heatmap = tf.reduce_sum(tf.multiply(weights[:, :, tf.newaxis, tf.newaxis], conv_outputs), axis=3)
        heatmap = tf.squeeze(heatmap, axis=0)

        # Apply ReLU and normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    def overlay_heatmap(
        self,
        input_data: np.ndarray,
        original_spectrogram: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay heatmap on original spectrogram.

        Args:
            input_data: Model input
            original_spectrogram: Original spectrogram (height, width)
            alpha: Transparency of heatmap overlay

        Returns:
            Overlaid visualization
        """
        heatmap = self.compute_heatmap(input_data)

        # Resize heatmap to match spectrogram
        heatmap_resized = tf.image.resize(
            heatmap[tf.newaxis, :, :, tf.newaxis],
            (original_spectrogram.shape[0], original_spectrogram.shape[1]),
        ).numpy()
        heatmap_resized = np.squeeze(heatmap_resized)

        # Normalize spectrogram for visualization
        spec_normalized = (original_spectrogram - original_spectrogram.min()) / (
            original_spectrogram.max() - original_spectrogram.min() + 1e-8
        )

        # Blend
        overlaid = (1 - alpha) * spec_normalized + alpha * heatmap_resized

        return overlaid


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model explainability.
    
    Shows which input features contribute most to the deepfake decision.
    """

    def __init__(self, model, background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: Keras model
            background_data: Background data for baseline (uses zeros if None)
        """
        self.model = model
        if background_data is None:
            # Use zeros as baseline
            input_shape = model.input_shape
            background_data = np.zeros((1,) + input_shape[1:])

        self.background_data = background_data
        logger.info("Initialized SHAP explainer")

    def explain_prediction(
        self,
        input_data: np.ndarray,
        n_samples: int = 100,
    ) -> Dict:
        """
        Generate SHAP explanation for a prediction.

        Args:
            input_data: Input to explain
            n_samples: Number of samples for estimation

        Returns:
            Dictionary with SHAP values and statistics
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Please install SHAP: pip install shap")

        # Create prediction function
        def predict_fn(x):
            return self.model.predict(x, verbose=0)

        # Create explainer
        explainer = shap.DeepExplainer(predict_fn, self.background_data)

        # Calculate SHAP values
        shap_values = explainer.shap_values(input_data)

        # Compute statistics
        feature_importance = np.abs(shap_values[0]).mean(axis=-1)  # Average across channels

        explanation = {
            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            "feature_importance": feature_importance.tolist(),
            "prediction": float(self.model.predict(input_data, verbose=0)[0, 0]),
        }

        logger.info("SHAP explanation computed")
        return explanation

    def plot_feature_importance(
        self,
        input_data: np.ndarray,
        output_path: str = "shap_importance.png",
    ):
        """
        Plot feature importance.

        Args:
            input_data: Input to explain
            output_path: Path to save plot
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Please install shap and matplotlib")

        # Get SHAP values
        def predict_fn(x):
            return self.model.predict(x, verbose=0)

        explainer = shap.DeepExplainer(predict_fn, self.background_data)
        shap_values = explainer.shap_values(input_data)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(shap_values[0].mean(axis=0)), cmap="hot", aspect="auto")
        plt.colorbar(label="Mean |SHAP value|")
        plt.title("Feature Importance (SHAP)")
        plt.xlabel("Frequency")
        plt.ylabel("Time Steps")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"SHAP importance plot saved to {output_path}")


class IntegratedGradients:
    """
    Integrated Gradients for feature importance attribution.
    
    Shows the contribution of each input feature to the model prediction.
    """

    def __init__(self, model):
        """
        Initialize Integrated Gradients explainer.

        Args:
            model: Keras/TensorFlow model
        """
        self.model = model
        logger.info("Initialized Integrated Gradients explainer")

    def integrated_gradients(
        self,
        input_data: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        steps: int = 50,
    ) -> np.ndarray:
        """
        Compute integrated gradients.

        Args:
            input_data: Input features
            baseline: Baseline input (zeros if None)
            steps: Number of integration steps

        Returns:
            Integrated gradient attribution
        """
        if baseline is None:
            baseline = np.zeros_like(input_data)

        # Generate interpolated inputs
        alphas = np.linspace(0, 1, steps)
        interpolated_inputs = []

        for alpha in alphas:
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated_inputs.append(interpolated)

        interpolated_inputs = np.concatenate(interpolated_inputs, axis=0)

        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(tf.Variable(interpolated_inputs))
            predictions = self.model(interpolated_inputs)

        gradients = tape.gradient(predictions, interpolated_inputs)

        # Average gradients
        avg_gradients = np.mean(gradients, axis=0)

        # Scale by (input - baseline)
        integrated_grads = (input_data - baseline) * avg_gradients

        logger.info(f"Integrated gradients computed: {integrated_grads.shape}")
        return integrated_grads

    def attribution_summary(
        self,
        input_data: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate attribution summary.

        Args:
            input_data: Input features
            baseline: Baseline for comparison

        Returns:
            Summary of important features
        """
        ig = self.integrated_gradients(input_data, baseline)

        # Compute importance
        importance = np.abs(ig).mean(axis=(0, 2))  # Average across batch and time

        summary = {
            "channel_importance": importance.tolist(),
            "top_channels": int(np.argmax(importance)),
            "max_importance": float(np.max(importance)),
            "mean_importance": float(np.mean(importance)),
        }

        return summary


class XAIVisualizer:
    """
    Unified XAI visualization module.
    
    Combines multiple explanation techniques for comprehensive interpretability.
    """

    def __init__(self, model, layer_name: str = "conv2d"):
        """
        Initialize XAI visualizer.

        Args:
            model: Keras model
            layer_name: Layer to visualize with Grad-CAM
        """
        self.model = model
        self.gradcam = GradCAM(model, layer_name)
        self.integrated_grads = IntegratedGradients(model)
        self.shap_explainer = SHAPExplainer(model)
        logger.info("Initialized unified XAI visualizer")

    def explain_prediction(
        self,
        input_data: np.ndarray,
        original_spectrogram: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate comprehensive explanation.

        Args:
            input_data: Model input
            original_spectrogram: Original spectrogram for visualization

        Returns:
            Complete explanation with multiple methods
        """
        prediction = self.model.predict(input_data, verbose=0)[0, 0]
        is_deepfake = prediction >= 0.5

        explanation = {
            "prediction": {
                "probability": float(prediction),
                "is_deepfake": bool(is_deepfake),
                "confidence": float(max(prediction, 1 - prediction)),
            },
            "gradcam_heatmap": self.gradcam.compute_heatmap(input_data).tolist(),
            "integrated_gradients": self.integrated_grads.attribution_summary(input_data),
        }

        # Add Grad-CAM overlay if spectrogram provided
        if original_spectrogram is not None:
            overlay = self.gradcam.overlay_heatmap(input_data, original_spectrogram)
            explanation["gradcam_overlay"] = overlay.tolist()

        logger.info("Comprehensive XAI explanation generated")
        return explanation

    def save_explanation(
        self,
        explanation: Dict,
        output_path: str = "explanation.json",
    ):
        """
        Save explanation to file.

        Args:
            explanation: Explanation dictionary
            output_path: Path to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(explanation, f, indent=2)

        logger.info(f"Explanation saved to {output_path}")


class SaliencyMap:
    """
    Generate saliency maps showing pixel-level importance.
    """

    def __init__(self, model):
        """
        Initialize saliency map generator.

        Args:
            model: Keras model
        """
        self.model = model
        logger.info("Initialized saliency map generator")

    def compute_saliency(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute saliency map.

        Args:
            input_data: Input features

        Returns:
            Saliency map showing important regions
        """
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor)
            loss = predictions[:, 0]

        # Compute gradients
        gradients = tape.gradient(loss, input_tensor)

        # Compute saliency
        saliency = tf.reduce_max(tf.abs(gradients), axis=1)

        return saliency.numpy()
