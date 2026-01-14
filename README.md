# Audio Deepfake Detection System

A production-ready deep learning system for detecting deepfakes in audio using a hybrid CNN-LSTM architecture with self-attention mechanisms.

## Features

- **Multi-scale CNN**: Captures spectral patterns at different scales (3x3, 5x5, 7x7 kernels)
- **Self-Attention Mechanism**: Weights temporal dependencies across feature scales
- **Bidirectional LSTM**: Models sequential audio patterns
- **Comprehensive Preprocessing**: Extracts MFCCs, Mel-spectrograms, and multiple audio features
- **Advanced Evaluation**: Includes accuracy, precision, recall, F1-score, ROC-AUC metrics
- **Cross-Validation**: K-fold cross-validation for robust model assessment
- **Production Inference**: Batch processing with configurable threshold-based detection
- **Configuration Management**: YAML/JSON-based configuration system

## Architecture

### Algorithm Steps

1. **Audio Preprocessing**
   - Load and resample audio to 16kHz
   - Extract MFCC features with delta and delta-delta
   - Compute log-mel spectrograms
   - Normalize and pad/truncate to fixed length (256 timesteps)

2. **Feature Enhancement**
   - Multi-scale CNN branches (3×3, 5×5, 7×7 kernels)
   - Concatenate multi-scale features
   - Additional convolution for feature enhancement

3. **Temporal Modeling**
   - Multi-head self-attention for temporal dependency weighting
   - Bidirectional LSTM for sequence modeling
   - Global average pooling over time

4. **Classification**
   - Dense layers with batch normalization and dropout
   - Binary sigmoid output for deepfake probability
   - Threshold-based classification (default: 0.5)

## Installation

### Requirements
- Python 3.8+
- TensorFlow 2.13+
- librosa
- scikit-learn
- numpy, scipy

### Setup

```bash
cd audio-deepfake-detection
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Training

```python
from src.preprocessing import AudioProcessor
from src.models import HybridDeepfakeDetector
from src.training import Trainer
from src.utils import setup_logging, Config

# Setup logging
setup_logging()

# Create model
model = HybridDeepfakeDetector(input_shape=(2, 39, 256))

# Train
trainer = Trainer(model, model_dir="models", log_dir="logs")
history = trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

# Save model
trainer.save_model("models/deepfake_detector.keras")
```

### 2. Audio Preprocessing

```python
from src.preprocessing import AudioProcessor

processor = AudioProcessor()

# Process single audio file
features = processor.process_audio("path/to/audio.wav")
# Output shape: (2, 39, 256)

# Batch processing
batch_features, valid_paths = processor.batch_process(["audio1.wav", "audio2.wav"])
# Output shape: (2, 2, 39, 256)
```

### 3. Inference

```python
from src.inference import DeepfakeDetector

detector = DeepfakeDetector(
    model_path="models/deepfake_detector.keras",
    threshold=0.5
)

# Single file detection
result = detector.detect_single("audio.wav")
print(f"Deepfake: {result['is_deepfake']}, Prob: {result['probability']:.4f}")

# Batch detection
batch_results = detector.detect_batch(["audio1.wav", "audio2.wav"])
print(f"Deepfakes detected: {batch_results['summary']['deepfakes_detected']}")
```

### 4. Cross-Validation

```python
from src.training import Trainer
from src.models import HybridDeepfakeDetector

model = HybridDeepfakeDetector()
trainer = Trainer(model)

cv_results = trainer.cross_validate(
    X=X_data,
    y=y_data,
    n_splits=5,
    epochs=30,
    batch_size=32
)
```

## Configuration

```python
from src.utils import Config

config = Config("config.yaml")
target_sr = config.get("audio.target_sr")
config.set("training.epochs", 100)
config.save_config("config_updated.yaml")
```

## Project Structure

```
audio-deepfake-detection/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hybrid_model.py
│   │   └── attention.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logger.py
│   └── __init__.py
├── data/
├── models/
├── logs/
├── tests/
├── requirements.txt
├── setup.py
├── config.yaml
└── README.md
```

## Model Architecture

**Input**: (batch_size, 2, 39, 256)
- Channel 0: MFCC features (13) + deltas (13) + delta-deltas (13)
- Channel 1: Log-mel spectrogram (128 bins)
- Timespan: 256 timesteps ≈ 8.5 seconds at 16kHz

**Processing Pipeline**:
1. **Multi-scale CNN**: 3×3, 5×5, 7×7 kernels (32 filters each)
2. **Self-Attention**: 8 heads for temporal weighting
3. **Bidirectional LSTM**: 128 units for sequence modeling
4. **Dense Layers**: 256 → 128 → 64 units with batch norm and dropout
5. **Output**: Sigmoid for binary classification

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Sensitivity, Specificity
- Confusion Matrix (TP, TN, FP, FN)

## Advanced Usage

```python
# Custom model parameters
model = HybridDeepfakeDetector(
    input_shape=(2, 39, 256),
    num_cnn_filters=64,
    lstm_units=256,
    dropout_rate=0.4,
    num_attention_heads=16
)

# Threshold optimization
optimal_threshold, f1 = MetricsCalculator.find_optimal_threshold(
    y_true=y_val,
    y_pred_proba=predictions
)
detector.set_threshold(optimal_threshold)

# Export results
detector.export_results(batch_results, "results.json")
```

## References

- **MFCC**: Mel-Frequency Cepstral Coefficients for audio feature extraction
- **Multi-scale CNN**: Capture spectral patterns at multiple scales
- **Self-Attention**: Transformer mechanism for temporal dependency weighting
- **LSTM**: Long Short-Term Memory for sequential pattern modeling
- **ROC-AUC**: Standard metric for binary classification performance

## License

This project is provided as-is for research and development purposes.
