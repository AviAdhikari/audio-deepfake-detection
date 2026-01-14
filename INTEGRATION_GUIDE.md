# Integration Guide: New AI Features with Existing System

## Overview

All new AI features integrate seamlessly with the existing audio deepfake detection system. This guide shows how to use them together.

## Component Integration Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO INPUT                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐           ┌──────────────────────┐
│  Traditional     │           │  Foundation Models    │
│  Preprocessing   │           │  (NEW)               │
│                  │           │                      │
│ • MFCC           │           │ • Wav2Vec2           │
│ • Mel-Spectrogram│           │ • HuBERT             │
│ • STFT           │           │ • Whisper            │
│ • Normalization  │           │ • AudioMAE           │
│                  │           │ • Ensemble           │
└────────┬─────────┘           └──────────┬───────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐           ┌──────────────────────┐
│  Original Models │           │  Transformer Models   │
│                  │           │  (NEW)               │
│ • Hybrid CNN-LSTM│           │                      │
│                  │           │ • Transformer        │
│                  │           │   Deepfake Detector  │
│                  │           │ • Hybrid Transformer │
│                  │           │   CNN Detector       │
└────────┬─────────┘           └──────────┬───────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐           ┌──────────────────────┐
│  Training        │           │  XAI & Visualization │
│  (Existing)      │           │  (NEW)               │
│                  │           │                      │
│ • Trainer        │           │ • Grad-CAM           │
│ • Metrics        │           │ • SHAP               │
│ • Cross-Val      │           │ • Integrated Grads   │
│ • Callbacks      │           │ • Saliency Maps      │
│                  │           │ • XAIVisualizer      │
└────────┬─────────┘           └──────────┬───────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
                    ┌──────────────┐
                    │   RESULTS    │
                    │ & PREDICTIONS│
                    └──────────────┘
```

## Usage Patterns

### Pattern 1: Transformer with Traditional Features

```python
from src.preprocessing.audio_processor import AudioProcessor
from src.models.transformer_model import TransformerDeepfakeDetector
from src.training.trainer import Trainer
from src.utils.config import ConfigManager

# Step 1: Preprocess audio (traditional)
processor = AudioProcessor()
X = processor.batch_process(audio_list)  # (N, 2, 39, 256)

# Step 2: Create transformer model (NEW)
model = TransformerDeepfakeDetector(
    input_shape=(2, 39, 256),
    num_transformer_blocks=4
)

# Step 3: Train with existing trainer
config = ConfigManager()
trainer = Trainer(config)
history, best_model = trainer.train(model, X_train, y_train, X_val, y_val)

# Step 4: Evaluate with existing metrics
from src.training.metrics import MetricsCalculator
metrics = MetricsCalculator.calculate_metrics(y_val, predictions)
```

**When to use**: When you want better temporal modeling but keep traditional features.

---

### Pattern 2: Traditional Model with Foundation Features

```python
from src.models.foundation_models import Wav2Vec2FeatureExtractor
from src.models.hybrid_model import HybridDeepfakeDetector
from src.training.trainer import Trainer

# Step 1: Extract foundation model features (NEW)
wav2vec2 = Wav2Vec2FeatureExtractor()
X_foundation = np.array([
    wav2vec2.extract_features(audio, sr=16000)
    for audio in audio_list
])  # (N, T, 768)

# Note: Need to reshape to match model input expectations
# Option A: Use as-is with custom model
# Option B: Average pool to get (N, 768)
X_pooled = X_foundation.mean(axis=1)

# Step 2: Use traditional model (or create custom model for this shape)
model = HybridDeepfakeDetector(input_shape=X_pooled[0].shape)

# Step 3: Train with existing trainer
trainer = Trainer(config)
history, best_model = trainer.train(model, X_pooled, y_train)
```

**When to use**: When you want self-supervised features with existing architecture.

---

### Pattern 3: Ensemble Foundation Models with Transformer

```python
from src.models.foundation_models import FoundationModelEnsemble
from src.models.transformer_model import HybridTransformerCNNDetector
from src.training.trainer import Trainer

# Step 1: Extract ensemble foundation features (NEW)
ensemble = FoundationModelEnsemble(
    model_names=["wav2vec2", "hubert", "whisper"]
)
X_ensemble = np.array([
    ensemble.extract_features(audio, sr=16000)
    for audio in audio_list
])  # (N, T, 2304) - concatenated features

# Step 2: Create custom transformer model for ensemble features
# Option: Create wrapper model that reshapes ensemble features
# For now, use average pooling
X_ensemble_pooled = X_ensemble.mean(axis=1)  # (N, 2304)

# Step 3: Use transformer with foundation features
model = HybridTransformerCNNDetector(input_shape=(2, 39, 256))

# Note: May need custom preprocessing to fit ensemble features
# into the expected shape, or create custom model

# Step 4: Train
trainer = Trainer(config)
history = trainer.train(model, X_data, y_train)
```

**When to use**: Best generalization for multiple deepfake types.

---

### Pattern 4: Inference with Explainability

```python
from src.preprocessing.audio_processor import AudioProcessor
from src.inference.detector import DeepfakeDetector
from src.xai.interpretability import XAIVisualizer

# Step 1: Load trained model
detector = DeepfakeDetector(
    model_path="models/best_detector.keras",
    threshold=0.5
)

# Step 2: Preprocess audio (existing)
processor = AudioProcessor()
features = processor.process_audio("test_audio.wav")

# Step 3: Make prediction (existing)
result = detector.detect_single("test_audio.wav")

# Step 4: Get explanation (NEW)
model = detector.model  # Get underlying Keras model
xai = XAIVisualizer(model, layer_name="conv2d")

# Get comprehensive explanation
mel_spec = processor.extract_mel_spectrogram("test_audio.wav")
explanation = xai.explain_prediction(features, original_spectrogram=mel_spec)

# Step 5: Present results
print(f"Prediction: {result['is_deepfake']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Explanation saved: {explanation}")
```

**When to use**: Production inference with interpretability requirements.

---

### Pattern 5: Error Analysis with XAI

```python
from src.inference.detector import DeepfakeDetector
from src.xai.interpretability import XAIVisualizer, SHAPExplainer

# Step 1: Get predictions on validation set
detector = DeepfakeDetector(model_path="models/detector.keras")
predictions = detector.detect_batch(X_val_audio_list)

# Step 2: Find errors
false_positives = np.where(
    (predictions['predictions'] >= 0.5) & (y_val == 0)
)[0]

# Step 3: Analyze with XAI
model = detector.model
xai = XAIVisualizer(model)

for fp_idx in false_positives[:5]:  # Analyze first 5 false positives
    X_fp = X_val[fp_idx:fp_idx+1]
    explanation = xai.explain_prediction(X_fp)
    
    print(f"False Positive {fp_idx}:")
    print(f"  Prediction: {explanation['prediction']['probability']:.4f}")
    print(f"  Top channel: {explanation['integrated_gradients']['top_channels']}")
    
    # Use SHAP for detailed analysis
    shap = SHAPExplainer(model, background_data=X_val[:100])
    shap_exp = shap.explain_prediction(X_fp)
```

**When to use**: Debugging and improving model performance.

---

### Pattern 6: Cross-Dataset Evaluation

```python
from src.models.foundation_models import FoundationModelEnsemble
from src.preprocessing.audio_processor import AudioProcessor
from src.training.trainer import Trainer

# Step 1: Extract features from dataset A (training)
processor = AudioProcessor()
X_train_foundation = np.array([
    FoundationModelEnsemble().extract_features(audio)
    for audio in train_audio_list
])

# Step 2: Train model on dataset A
model = TransformerDeepfakeDetector()
trainer = Trainer(config)
trainer.train(model, X_train_foundation, y_train)

# Step 3: Evaluate on dataset B (different deepfake source)
X_test_foundation = np.array([
    FoundationModelEnsemble().extract_features(audio)
    for audio in test_audio_list
])

# Step 4: Get predictions
predictions = model.predict(X_test_foundation)

# Step 5: Evaluate with existing metrics
from src.training.metrics import MetricsCalculator
metrics = MetricsCalculator.calculate_metrics(y_test, predictions)
print(f"Cross-dataset F1: {metrics['f1_score']:.4f}")
```

**When to use**: Evaluating generalization to new deepfake types.

---

## Integration Checklist

### For Using Transformers with Existing Code

- [x] Transformer models have same input shape as original (2, 39, 256)
- [x] Compatible with existing Trainer class
- [x] Compatible with existing metrics evaluation
- [x] Compatible with existing inference pipeline
- [x] Compatible with existing cross-validation

### For Using Foundation Models

- [ ] Verify output shape matches your model's input
- [ ] Consider time-averaging for dimension reduction
- [ ] Test memory usage with ensemble models
- [ ] Download models first (1GB+ for Whisper)
- [ ] Consider caching features for repeated use

### For Using XAI

- [x] Works with any Keras model (old or new)
- [x] Can visualize predictions from trained models
- [x] Can be applied post-training (no retraining needed)
- [x] Graceful degradation if SHAP not installed
- [x] Handles model predictions from any detector

## API Compatibility

### Existing APIs Still Work

```python
# AudioProcessor (unchanged)
processor.process_audio("audio.wav")
processor.batch_process(audio_list)
processor.extract_mfcc()
processor.extract_mel_spectrogram()

# Trainer (unchanged)
trainer.train(model, X_train, y_train, X_val, y_val)
trainer.cross_validate(X, y, n_splits=5)
trainer.evaluate(model, X_test, y_test)

# DeepfakeDetector (unchanged)
detector.detect_single("audio.wav")
detector.detect_batch(audio_list)
detector.export_results()

# MetricsCalculator (unchanged)
MetricsCalculator.calculate_metrics(y_true, y_pred)
MetricsCalculator.find_optimal_threshold()
```

### New APIs for Features

```python
# Foundation Models
wav2vec2 = Wav2Vec2FeatureExtractor()
features = wav2vec2.extract_features(audio, sr=16000)

# Transformers
model = TransformerDeepfakeDetector(input_shape=(2, 39, 256))

# XAI
xai = XAIVisualizer(model)
explanation = xai.explain_prediction(input_data)
```

## Data Flow Examples

### Example 1: Traditional Pipeline
```
Audio → AudioProcessor → (2,39,256) → HybridDeepfakeDetector 
→ Predictions → MetricsCalculator → Results
```

### Example 2: Foundation Model Pipeline
```
Audio → Wav2Vec2FeatureExtractor → (1,T,768) → Custom Model 
→ Predictions → MetricsCalculator → Results
```

### Example 3: Transformer Pipeline
```
Audio → AudioProcessor → (2,39,256) → TransformerDeepfakeDetector 
→ Predictions → XAIVisualizer → Explanation
```

### Example 4: Ensemble Pipeline
```
Audio → FoundationModelEnsemble → (1,T,2304) → HybridTransformerCNN 
→ Predictions → SHAPExplainer → Feature Importance
```

## Performance Recommendations

### Fast Inference
1. Use original HybridDeepfakeDetector (CNN-LSTM)
2. Use traditional AudioProcessor features
3. Batch inference when possible

### Best Accuracy
1. Use TransformerDeepfakeDetector
2. Use FoundationModelEnsemble for features
3. Use HybridTransformerCNNDetector as alternative

### Production with Explainability
1. Use any model (depends on requirements)
2. Prepare Grad-CAM heatmaps post-inference
3. Cache SHAP explanations (slower computation)

### Development/Debugging
1. Use any model
2. Use full XAI (Grad-CAM + SHAP + Integrated Gradients)
3. Use foundation models for better understanding

## Common Integration Issues & Solutions

### Issue: Shape Mismatch with Foundation Features
**Solution**: Use time-averaging or custom reshaping layer
```python
X = foundation_features  # (N, T, 768)
X_pooled = X.mean(axis=1)  # (N, 768)
```

### Issue: Memory Too High with Ensemble
**Solution**: Use single foundation model or reduce batch size
```python
# Instead of ensemble
ensemble = FoundationModelEnsemble(model_names=["wav2vec2", "hubert", "whisper"])

# Use single model
single = Wav2Vec2FeatureExtractor()
```

### Issue: Slow SHAP Computation
**Solution**: Use Grad-CAM or Integrated Gradients instead
```python
# Faster
gradcam = GradCAM(model, layer_name="conv2d")
heatmap = gradcam.compute_heatmap(X)

# Alternatively
ig = IntegratedGradients(model)
attribution = ig.integrated_gradients(X)
```

## Summary

The new AI features integrate seamlessly through:

1. **Consistent Input Shapes**: (2, 39, 256) or compatible variants
2. **Compatible Training**: All models work with Trainer class
3. **Unified Inference**: All models work with DeepfakeDetector
4. **Post-Training XAI**: Works with any model without retraining
5. **Modular Design**: Use any combination of components

This allows you to:
- ✅ Keep existing code unchanged
- ✅ Add new features incrementally
- ✅ Mix old and new approaches
- ✅ Experiment with multiple combinations
- ✅ Scale from simple to complex setups

Ready to integrate! Start with one pattern and expand as needed.
