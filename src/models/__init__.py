"""Neural network models for deepfake detection."""

from .hybrid_model import HybridDeepfakeDetector
from .attention import MultiHeadAttention

__all__ = ["HybridDeepfakeDetector", "MultiHeadAttention"]
