
from mcl.machine_types import intp, f32
from mcl.ndarray import Array
import random
import math
from mcl.vm import _get_machine_value
from mcl.array_math import array_exp, array_sum, array_max, array_matmul, array_maximum

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = array_exp(x - array_max(x, axis=axis, keepdims=True))
    return e_x / array_sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(query, key, value):
    """
    Compute 'Scaled Dot Product Attention'
    Args:
        query: (batch_size, sequence_length, embedding_dim)
        key: (batch_size, sequence_length, embedding_dim)
        value: (batch_size, sequence_length, embedding_dim)
    Returns:
        context: (batch_size, sequence_length, embedding_dim)
        weights: (batch_size, sequence_length, sequence_length)
    """
    # Reshape for proper matrix multiplication
    Q = query.reshape((intp(-1), query.shape[1], query.shape[2]))
    K = key.reshape((intp(-1), key.shape[1], key.shape[2]))
    V = value.reshape((intp(-1), value.shape[1], value.shape[2]))

    # Scale dot product attention
    d_k = query.shape[-1]
    scores = array_matmul(Q, K.transpose((intp(0), intp(2), intp(1)))) / f32(math.sqrt(_get_machine_value(d_k)))

    # Get attention weights
    weights = softmax(scores, axis=-1)

    # Calculate weighted sum
    context = array_matmul(weights, V)

    return context.reshape(query.shape), weights

class MultiHeadAttention:
    def __init__(self, num_heads=intp(8), embedding_dim=intp(128)):
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

    def split_heads(self, x):
        """Split the last dimension into (heads, depth)"""
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        x = x.reshape((batch_size, sequence_length, self.num_heads, intp(-1)))
        return x.transpose((intp(0), intp(2), intp(1), intp(3)))

    def combine_heads(self, x):
        """Combine heads dimension"""
        batch_size = x.shape[0]
        sequence_length = x.shape[2]
        x = x.transpose((intp(0), intp(2), intp(1), intp(3))).reshape((batch_size, sequence_length, intp(-1)))
        return x

    def forward(self, query, key, value):
        # Split heads
        q = self.split_heads(query)
        k = self.split_heads(key)
        v = self.split_heads(value)

        # Apply attention
        context, weights = scaled_dot_product_attention(q, k, v)

        # Combine heads
        return self.combine_heads(context), weights

class FeedForwardNetwork:
    def __init__(self, embedding_dim=intp(128), hidden_dim=intp(256)):
        self.W1 = Array.random((embedding_dim, hidden_dim))
        self.W2 = Array.random((hidden_dim, embedding_dim))

    def forward(self, x):
        return array_matmul(array_maximum(array_matmul(x, self.W1), f32(0)), self.W2)

class TransformerLayer:
    def __init__(self, num_heads=intp(8), embedding_dim=intp(128), dropout=0.1):
        self.self_attn = MultiHeadAttention(num_heads, embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim)
        self.dropout = dropout

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn.forward(x, x, x)

        # Feed-forward network
        ff_output = self.feed_forward.forward(attn_output + x)

        return ff_output

# Example usage
random.seed(42)

batch_size = intp(32)
sequence_length = intp(50)
embedding_dim = intp(128)

input_data: Array = Array.random((batch_size, sequence_length, embedding_dim))

transformer_layer = TransformerLayer(num_heads=intp(8), embedding_dim=embedding_dim)

output = transformer_layer.forward(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {output.shape}")
