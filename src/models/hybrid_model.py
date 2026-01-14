"""Hybrid CNN-LSTM model for audio deepfake detection."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple
from .attention import MultiHeadAttention


class HybridDeepfakeDetector(Model):
    """
    Hybrid deep learning model combining CNN, self-attention, and LSTM
    for audio deepfake detection.

    Architecture:
    1. Multi-scale CNN branches to capture spectral patterns
    2. Self-attention to weight temporal dependencies
    3. Bidirectional LSTM for sequence modeling
    4. Dense layers for binary classification
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (2, 39, 256),
        num_cnn_filters: int = 32,
        lstm_units: int = 128,
        dropout_rate: float = 0.3,
        num_attention_heads: int = 8,
        **kwargs
    ):
        """
        Initialize hybrid model.

        Args:
            input_shape: Input shape (channels, height, width)
            num_cnn_filters: Number of filters in CNN layers
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
            num_attention_heads: Number of attention heads
        """
        super().__init__(**kwargs)
        self.input_shape_config = input_shape
        self.num_cnn_filters = num_cnn_filters
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.num_attention_heads = num_attention_heads

        self._build_model()

    def _build_model(self):
        """Build the complete hybrid model architecture."""
        # Input layer
        inputs = keras.Input(shape=self.input_shape_config)

        # ===== Multi-scale CNN Branch =====
        # Scale 1: 3x3 kernels
        cnn1 = layers.Conv2D(
            self.num_cnn_filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )(inputs)
        cnn1 = layers.BatchNormalization()(cnn1)
        cnn1 = layers.MaxPooling2D(pool_size=(2, 2))(cnn1)
        cnn1 = layers.Dropout(self.dropout_rate)(cnn1)

        # Scale 2: 5x5 kernels
        cnn2 = layers.Conv2D(
            self.num_cnn_filters,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )(inputs)
        cnn2 = layers.BatchNormalization()(cnn2)
        cnn2 = layers.MaxPooling2D(pool_size=(2, 2))(cnn2)
        cnn2 = layers.Dropout(self.dropout_rate)(cnn2)

        # Scale 3: 7x7 kernels
        cnn3 = layers.Conv2D(
            self.num_cnn_filters,
            kernel_size=(7, 7),
            padding="same",
            activation="relu",
        )(inputs)
        cnn3 = layers.BatchNormalization()(cnn3)
        cnn3 = layers.MaxPooling2D(pool_size=(2, 2))(cnn3)
        cnn3 = layers.Dropout(self.dropout_rate)(cnn3)

        # Concatenate multi-scale features
        cnn_concat = layers.Concatenate()([cnn1, cnn2, cnn3])

        # Additional convolutional layer to enhance features
        cnn_enhanced = layers.Conv2D(
            self.num_cnn_filters * 2,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )(cnn_concat)
        cnn_enhanced = layers.BatchNormalization()(cnn_enhanced)
        cnn_enhanced = layers.Dropout(self.dropout_rate)(cnn_enhanced)

        # Reshape for sequence modeling (batch, time_steps, features)
        # Flatten spatial dimensions
        batch_size = tf.shape(cnn_enhanced)[0]
        height = cnn_enhanced.shape[1]
        width = cnn_enhanced.shape[2]
        channels = cnn_enhanced.shape[3]

        cnn_reshaped = layers.Reshape(
            (width, height * channels)
        )(cnn_enhanced)

        # ===== Self-Attention on Temporal Dependencies =====
        attn_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=64,
        )(cnn_reshaped)

        # ===== Bidirectional LSTM for Sequence Modeling =====
        lstm_output = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
            )
        )(attn_output)

        # ===== Temporal Pooling =====
        # Global average pooling over time dimension
        pooled = layers.GlobalAveragePooling1D()(lstm_output)

        # ===== Dense Layers for Classification =====
        dense = layers.Dense(256, activation="relu")(pooled)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        dense = layers.Dense(128, activation="relu")(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        dense = layers.Dense(64, activation="relu")(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        # Binary classification output
        outputs = layers.Dense(1, activation="sigmoid")(dense)

        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name="HybridDeepfakeDetector")

    def call(self, inputs, training=False):
        """Forward pass."""
        return self.model(inputs, training=training)

    def get_config(self):
        """Get model configuration."""
        return {
            "input_shape": self.input_shape_config,
            "num_cnn_filters": self.num_cnn_filters,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "num_attention_heads": self.num_attention_heads,
        }

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)

    def summary_detailed(self):
        """Print detailed model summary."""
        return self.model.summary()
