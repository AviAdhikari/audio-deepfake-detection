"""Transformer-based models for audio deepfake detection."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple
import numpy as np


class TransformerBlock(layers.Layer):
    """Transformer encoder block with multi-head attention."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1, **kwargs):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            rate: Dropout rate
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=rate)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        """Apply transformer block."""
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        """Get layer configuration."""
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        }


class TransformerDeepfakeDetector(Model):
    """
    Transformer-based model for audio deepfake detection.
    
    Replaces LSTM with Transformer encoders for superior temporal modeling
    and long-range dependency capture.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (2, 39, 256),
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 256,
        num_transformer_blocks: int = 4,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        """
        Initialize Transformer model.

        Args:
            input_shape: Input shape (channels, height, width)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            dropout_rate: Dropout rate
        """
        super().__init__(**kwargs)
        self.input_shape_config = input_shape
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate

        self._build_model()

    def _build_model(self):
        """Build the transformer model."""
        inputs = keras.Input(shape=self.input_shape_config)

        # ===== CNN Feature Extraction =====
        # Multi-scale CNN to extract local patterns
        cnn1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        cnn1 = layers.BatchNormalization()(cnn1)
        cnn1 = layers.MaxPooling2D((2, 2))(cnn1)
        cnn1 = layers.Dropout(self.dropout_rate)(cnn1)

        cnn2 = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(inputs)
        cnn2 = layers.BatchNormalization()(cnn2)
        cnn2 = layers.MaxPooling2D((2, 2))(cnn2)
        cnn2 = layers.Dropout(self.dropout_rate)(cnn2)

        cnn3 = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(inputs)
        cnn3 = layers.BatchNormalization()(cnn3)
        cnn3 = layers.MaxPooling2D((2, 2))(cnn3)
        cnn3 = layers.Dropout(self.dropout_rate)(cnn3)

        cnn_concat = layers.Concatenate()([cnn1, cnn2, cnn3])
        cnn_enhanced = layers.Conv2D(
            self.embed_dim,
            (3, 3),
            padding="same",
            activation="relu",
        )(cnn_concat)
        cnn_enhanced = layers.BatchNormalization()(cnn_enhanced)
        cnn_enhanced = layers.Dropout(self.dropout_rate)(cnn_enhanced)

        # ===== Reshape to Sequence =====
        batch_size = tf.shape(cnn_enhanced)[0]
        height = cnn_enhanced.shape[1]
        width = cnn_enhanced.shape[2]
        channels = cnn_enhanced.shape[3]

        cnn_reshaped = layers.Reshape((width, height * channels))(cnn_enhanced)

        # Project to embedding dimension if needed
        if height * channels != self.embed_dim:
            cnn_reshaped = layers.Dense(self.embed_dim)(cnn_reshaped)

        # ===== Transformer Encoder Stack =====
        x = cnn_reshaped
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                rate=self.dropout_rate,
            )(x)

        # ===== Global Pooling =====
        pooled = layers.GlobalAveragePooling1D()(x)

        # ===== Classification Head =====
        dense = layers.Dense(256, activation="relu")(pooled)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        dense = layers.Dense(128, activation="relu")(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        dense = layers.Dense(64, activation="relu")(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        outputs = layers.Dense(1, activation="sigmoid")(dense)

        self.model = Model(inputs=inputs, outputs=outputs, name="TransformerDeepfakeDetector")

    def call(self, inputs, training=False):
        """Forward pass."""
        return self.model(inputs, training=training)

    def get_config(self):
        """Get model configuration."""
        return {
            "input_shape": self.input_shape_config,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "dropout_rate": self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)


class HybridTransformerCNNDetector(Model):
    """
    Hybrid model combining Transformer and CNN for deepfake detection.
    
    Uses Transformer for global context modeling with CNN for local
    spectral pattern detection.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (2, 39, 256),
        num_cnn_filters: int = 32,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_transformer_blocks: int = 2,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        """Initialize hybrid model."""
        super().__init__(**kwargs)
        self.input_shape_config = input_shape
        self.num_cnn_filters = num_cnn_filters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate

        self._build_model()

    def _build_model(self):
        """Build hybrid model."""
        inputs = keras.Input(shape=self.input_shape_config)

        # ===== Multi-scale CNN Branch =====
        cnn1 = layers.Conv2D(self.num_cnn_filters, (3, 3), padding="same", activation="relu")(inputs)
        cnn1 = layers.BatchNormalization()(cnn1)
        cnn1 = layers.MaxPooling2D((2, 2))(cnn1)
        cnn1 = layers.Dropout(self.dropout_rate)(cnn1)

        cnn2 = layers.Conv2D(self.num_cnn_filters, (5, 5), padding="same", activation="relu")(inputs)
        cnn2 = layers.BatchNormalization()(cnn2)
        cnn2 = layers.MaxPooling2D((2, 2))(cnn2)
        cnn2 = layers.Dropout(self.dropout_rate)(cnn2)

        cnn3 = layers.Conv2D(self.num_cnn_filters, (7, 7), padding="same", activation="relu")(inputs)
        cnn3 = layers.BatchNormalization()(cnn3)
        cnn3 = layers.MaxPooling2D((2, 2))(cnn3)
        cnn3 = layers.Dropout(self.dropout_rate)(cnn3)

        cnn_concat = layers.Concatenate()([cnn1, cnn2, cnn3])
        cnn_enhanced = layers.Conv2D(self.embed_dim, (3, 3), padding="same", activation="relu")(cnn_concat)
        cnn_enhanced = layers.BatchNormalization()(cnn_enhanced)
        cnn_enhanced = layers.Dropout(self.dropout_rate)(cnn_enhanced)

        # ===== Reshape and Embed =====
        cnn_reshaped = layers.Reshape(
            (cnn_enhanced.shape[2], cnn_enhanced.shape[1] * cnn_enhanced.shape[3])
        )(cnn_enhanced)
        if cnn_enhanced.shape[1] * cnn_enhanced.shape[3] != self.embed_dim:
            cnn_reshaped = layers.Dense(self.embed_dim)(cnn_reshaped)

        # ===== Transformer Encoder =====
        x = cnn_reshaped
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.embed_dim * 2,
                rate=self.dropout_rate,
            )(x)

        # ===== Global Pooling =====
        pooled = layers.GlobalAveragePooling1D()(x)

        # ===== Classification =====
        dense = layers.Dense(256, activation="relu")(pooled)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        dense = layers.Dense(128, activation="relu")(dense)
        dense = layers.BatchNormalization()(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        dense = layers.Dense(64, activation="relu")(dense)
        dense = layers.Dropout(self.dropout_rate)(dense)

        outputs = layers.Dense(1, activation="sigmoid")(dense)

        self.model = Model(inputs=inputs, outputs=outputs, name="HybridTransformerCNNDetector")

    def call(self, inputs, training=False):
        """Forward pass."""
        return self.model(inputs, training=training)

    def get_config(self):
        """Get model configuration."""
        return {
            "input_shape": self.input_shape_config,
            "num_cnn_filters": self.num_cnn_filters,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_transformer_blocks": self.num_transformer_blocks,
            "dropout_rate": self.dropout_rate,
        }

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
