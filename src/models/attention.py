"""Multi-head self-attention mechanism."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadAttention(layers.Layer):
    """Multi-head self-attention layer for temporal dependency weighting."""

    def __init__(self, num_heads: int = 8, key_dim: int = 64, **kwargs):
        """
        Initialize multi-head attention layer.

        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of each attention head
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        """Build layer weights."""
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=0.1,
        )
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, x, training=False):
        """Apply multi-head attention."""
        # Self-attention: use same input for query, key, value
        attn_output = self.attention(x, x, training=training)
        # Add & Norm
        output = self.layernorm(x + attn_output)
        return output

    def get_config(self):
        """Get layer configuration."""
        return {
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        }
