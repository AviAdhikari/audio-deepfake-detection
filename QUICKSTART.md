"""Quick start guide for the audio deepfake detection system."""

# Audio Deepfake Detection - Quick Start Guide

## Installation

```bash
cd /workspaces/audio-deepfake-detection
pip install -r requirements.txt
```

## Project Components

### 1. Audio Preprocessing (`src/preprocessing/`)
- **AudioProcessor**: Handles audio loading, feature extraction, and preprocessing
- Extracts MFCC, Mel-spectrogram, STFT, Chroma, and Spectral Centroid features
- Normalizes and pads features to consistent dimensions

### 2. Neural Network Model (`src/models/`)
- **HybridDeepfakeDetector**: Main model combining CNN, LSTM, and Attention
- **MultiHeadAttention**: Self-attention layer for temporal weighting
- Multi-scale CNN (3×3, 5×5, 7×7 kernels)
- Bidirectional LSTM with 128 units
- Dense classification layers

### 3. Training Pipeline (`src/training/`)
- **Trainer**: Handles model training, validation, and evaluation
- **MetricsCalculator**: Computes accuracy, precision, recall, F1, ROC-AUC
- K-fold cross-validation support
- Model checkpointing and early stopping

### 4. Inference (`src/inference/`)
- **DeepfakeDetector**: Production-ready inference interface
- Single and batch detection
- Configurable classification threshold
- Result export to JSON

### 5. Utilities (`src/utils/`)
- **Config**: YAML/JSON configuration management
- **setup_logging**: Logging configuration

## Usage Examples

### Example 1: Audio Preprocessing

```python
from src.preprocessing import AudioProcessor

processor = AudioProcessor(
    target_sr=16000,
    n_mels=128,
    n_mfcc=13
)

# Process single file
features = processor.process_audio("path/to/audio.wav")
print(features.shape)  # (2, 39, 256)

# Batch processing
batch, paths = processor.batch_process(["audio1.wav", "audio2.wav"])
print(batch.shape)  # (2, 2, 39, 256)
```

### Example 2: Model Training

```python
from src.models import HybridDeepfakeDetector
from src.training import Trainer

# Create model
model = HybridDeepfakeDetector(
    input_shape=(2, 39, 256),
    num_cnn_filters=32,
    lstm_units=128,
    dropout_rate=0.3
)

# Train
trainer = Trainer(model, model_dir="models", log_dir="logs")
history = trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

# Save
trainer.save_model("models/deepfake_detector.keras")
```

### Example 3: Inference

```python
from src.inference import DeepfakeDetector

detector = DeepfakeDetector(
    model_path="models/deepfake_detector.keras",
    threshold=0.5
)

# Single detection
result = detector.detect_single("audio.wav")
print(f"Deepfake: {result['is_deepfake']}")
print(f"Probability: {result['probability']:.4f}")

# Batch detection
batch_results = detector.detect_batch(["audio1.wav", "audio2.wav"])
print(batch_results['summary'])
```

### Example 4: Cross-Validation

```python
from src.training import Trainer
from src.models import HybridDeepfakeDetector

trainer = Trainer(HybridDeepfakeDetector())
cv_results = trainer.cross_validate(
    X=X_data,
    y=y_data,
    n_splits=5,
    epochs=30
)

print(cv_results['average_metrics'])
```

## Configuration

Edit `config.yaml` to customize:
- Audio parameters (sampling rate, MFCC, mel-bands)
- Model architecture (filters, LSTM units, dropout)
- Training settings (epochs, batch size, learning rate)
- Paths (data, models, logs)

```yaml
audio:
  target_sr: 16000
  n_mels: 128
  target_length: 256

model:
  num_cnn_filters: 32
  lstm_units: 128
  dropout_rate: 0.3

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

## Running Examples

### Training Example
```bash
python examples/train_model.py --config config.yaml
```

### Inference Example
```bash
python examples/inference_example.py --model models/deepfake_detector.keras --audio audio.wav
```

### Preprocessing Example
```bash
python examples/preprocessing_example.py --audio audio.wav
```

### Cross-Validation Example
```bash
python examples/cross_validation_example.py
```

## Model Architecture Details

**Input Shape**: (batch_size, 2, 39, 256)
- Channel 0: MFCC (13) + delta (13) + delta-delta (13) = 39 features
- Channel 1: Mel-spectrogram (128 features)
- Time steps: 256 (≈ 8.5 seconds at 16kHz)

**Processing Pipeline**:
1. **Multi-scale CNN**
   - Branch 1: Conv2D(3×3, 32) → BatchNorm → MaxPool → Dropout
   - Branch 2: Conv2D(5×5, 32) → BatchNorm → MaxPool → Dropout
   - Branch 3: Conv2D(7×7, 32) → BatchNorm → MaxPool → Dropout
   - Concatenate: 96 filters
   - Enhance: Conv2D(3×3, 64) → BatchNorm → Dropout

2. **Reshape to Sequence**
   - Flatten spatial dimensions for temporal modeling

3. **Self-Attention**
   - MultiHeadAttention(8 heads, key_dim=64)
   - Layer normalization

4. **Bidirectional LSTM**
   - 128 units
   - Dropout and recurrent dropout (0.3)
   - Return sequences for temporal modeling

5. **Global Average Pooling**
   - Pool over time dimension

6. **Dense Classification**
   - Dense(256, relu) → BatchNorm → Dropout
   - Dense(128, relu) → BatchNorm → Dropout
   - Dense(64, relu) → Dropout
   - Dense(1, sigmoid) → Binary output

## Key Features

✓ **Multi-scale feature extraction**: Captures patterns at different temporal/spectral scales
✓ **Self-attention mechanism**: Learns which parts of the signal are important
✓ **Bidirectional LSTM**: Models forward and backward temporal dependencies
✓ **Comprehensive preprocessing**: MFCCs, Mel-spectrograms, and multiple audio features
✓ **Robust evaluation**: Accuracy, precision, recall, F1, ROC-AUC, sensitivity, specificity
✓ **Cross-validation**: K-fold CV for assessing generalization
✓ **Production ready**: Batch processing, threshold tuning, JSON export
✓ **Configurable**: YAML-based configuration system
✓ **Well-documented**: Extensive docstrings and usage examples

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: False positive rate control
- **Recall**: False negative rate control
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall discriminative ability
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Confusion Matrix**: TP, TN, FP, FN counts

## File Organization

```
audio-deepfake-detection/
├── src/
│   ├── preprocessing/        # Audio feature extraction
│   ├── models/              # Neural network architecture
│   ├── training/            # Training and evaluation
│   ├── inference/           # Production inference
│   └── utils/               # Configuration and logging
├── examples/                 # Usage examples
├── tests/                    # Unit tests
├── data/                     # Dataset directory
├── models/                   # Saved model directory
├── logs/                     # Training logs and history
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md                # Full documentation
```

## Tips for Best Results

1. **Data Quality**: Use high-quality audio samples
2. **Balanced Dataset**: Ensure equal classes (real vs deepfake)
3. **Augmentation**: Consider data augmentation for training
4. **Threshold Tuning**: Use `find_optimal_threshold()` for your dataset
5. **Cross-Validation**: Always validate with CV, not just train/val/test
6. **Batch Processing**: Use batch inference for multiple files
7. **GPU**: Enable GPU acceleration for faster training
8. **Monitoring**: Check TensorBoard logs during training

## Troubleshooting

**Out of Memory**:
- Reduce batch_size
- Reduce num_cnn_filters
- Process files individually

**Poor Performance**:
- Check audio quality and sample rate
- Verify labels are correct (0=real, 1=deepfake)
- Increase training epochs
- Try different learning rates (0.0001 to 0.01)

**Audio Loading Issues**:
- Ensure files are in supported format (WAV, MP3, FLAC, M4A)
- Check file permissions
- Verify audio mono or will be converted to mono

## References

- MFCC: Mel-Frequency Cepstral Coefficients (speech processing standard)
- Multi-scale CNN: Captures features at multiple scales
- Self-Attention: Transformer mechanism for temporal weighting
- LSTM: Long Short-Term Memory for sequence modeling
- Librosa: Python audio analysis library

## Contact & Support

For issues or questions, refer to:
- README.md for full documentation
- Docstrings in source code for API details
- Examples folder for usage patterns
