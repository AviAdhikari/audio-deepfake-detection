"""Package initialization."""

__version__ = "1.0.0"
__author__ = "Audio Deepfake Detection Team"

from src.preprocessing import AudioProcessor
from src.models import HybridDeepfakeDetector
from src.training import Trainer, MetricsCalculator
from src.inference import DeepfakeDetector
from src.utils import Config, setup_logging

__all__ = [
    "AudioProcessor",
    "HybridDeepfakeDetector",
    "Trainer",
    "MetricsCalculator",
    "DeepfakeDetector",
    "Config",
    "setup_logging",
]
