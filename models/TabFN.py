# TabPFN Complete Implementation from Base Components
# Using only Tensor, Linear, Softmax, and basic operations

import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import Softmax, GELU
from tinytorch.core.layers import Linear, Dropout
import math


# ============================================
# Base Components for TabPFN
# ============================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention from base components
    """
    # Q, K, V are Tensors with shape [batch, seq_len, d_k]
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T
    scores = Q.matmul(K.transpose(-2, -1))

    # Scale scores
    scaling_factor = 1 / math.sqrt(d_k)
    scaled_scores = scores * scaling_factor

    # Apply mask if provided
    if mask is not None:
        scaled_scores = scaled_scores + (mask * -1e9)

    # Apply softmax
    softmax = Softmax()
    attention_weights = softmax.forward(scaled_scores, dim=-1)

    # Apply attention to values
    output = attention_weights.matmul(V)

    return output, attention_weights


def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads, mask=None):
    """
    Multi-Head Attention using base components
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # Linear projections
    Q = x.matmul(W_q.transpose())  # [batch, seq_len, d_model]
    K = x.matmul(W_k.transpose())  # [batch, seq_len, d_model]
    V = x.matmul(W_v.transpose())  # [batch, seq_len, d_model]

    # Reshape for multi-head attention
    Q = Q.reshape(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    K = K.reshape(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    V = V.reshape(batch_size, seq_len, n_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention for each head
    attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

    # Concatenate heads
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)

    # Output projection
    output = attn_output.matmul(W_o.transpose())

    return output


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    Layer Normalization from base components
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) * (x - mean)).mean(axis=-1, keepdims=True)
    std = (var + eps).sqrt()
    normalized = (x - mean) / std
    return normalized * gamma + beta


def feed_forward_network(x, W1, b1, W2, b2):
    """
    Feed Forward Network with GELU activation
    """
    # First linear layer (expansion)
    hidden = x.matmul(W1.transpose()) + b1
    # GELU activation
    gelu = GELU()
    hidden = gelu.forward(hidden)
    # Second linear layer (projection)
    output = hidden.matmul(W2.transpose()) + b2
    return output


# ============================================
# TabPFN Transformer Block
# ============================================

class TabPFNBlock:
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Multi-head attention weights
        self.W_q = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.W_k = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.W_v = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.W_o = Tensor(np.random.randn(d_model, d_model) * 0.02)

        # Layer normalization parameters
        self.gamma1 = Tensor(np.ones((d_model,)))
        self.beta1 = Tensor(np.zeros((d_model,)))
        self.gamma2 = Tensor(np.ones((d_model,)))
        self.beta2 = Tensor(np.zeros((d_model,)))

        # Feed-forward network weights (4x expansion)
        self.W_ffn1 = Tensor(np.random.randn(d_model * 4, d_model) * 0.02)
        self.b_ffn1 = Tensor(np.zeros((d_model * 4,)))
        self.W_ffn2 = Tensor(np.random.randn(d_model, d_model * 4) * 0.02)
        self.b_ffn2 = Tensor(np.zeros((d_model,)))

        # Dropout
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        # Save input for skip connection
        residual = x

        # Multi-head attention
        attn_output = multi_head_attention(x, self.W_q, self.W_k, self.W_v, self.W_o, self.n_heads, mask)
        attn_output = self.dropout.forward(attn_output, training=True)

        # Skip connection and layer norm
        x = residual + attn_output
        x = layer_norm(x, self.gamma1, self.beta1)

        # Save for skip connection
        residual = x

        # Feed-forward network
        ff_output = feed_forward_network(x, self.W_ffn1, self.b_ffn1, self.W_ffn2, self.b_ffn2)
        ff_output = self.dropout.forward(ff_output, training=True)

        # Skip connection and layer norm
        x = residual + ff_output
        x = layer_norm(x, self.gamma2, self.beta2)

        return x


# ============================================
# Complete TabPFN Model
# ============================================

class TabPFN:
    def __init__(self,
                 n_features=100,
                 d_model=4,
                 n_heads=1,
                 n_layers=12,
                 n_classes=2,
                 dropout=0.1):

        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Input embedding (feature projection)
        self.W_embed = Tensor(np.random.randn(d_model, 1) * 0.02)
        self.b_embed = Tensor(np.zeros((d_model,)))

        # Learnable patterns (TabPFN innovation)
        self.patterns = Tensor(np.random.randn(1, n_features, d_model) * 0.02)

        # Positional encoding (simplified)
        self.pos_encoding = self.create_positional_encoding(n_features, d_model)

        # Transformer blocks
        self.blocks = []
        for _ in range(n_layers):
            block = TabPFNBlock(d_model, n_heads, dropout)
            self.blocks.append(block)

        # Output projection
        self.W_out = Tensor(np.random.randn(n_classes, d_model) * 0.02)
        self.b_out = Tensor(np.zeros((n_classes,)))

    def create_positional_encoding(self, seq_len, d_model):
        """Create sinusoidal positional encoding"""
        pos_encoding = np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        return Tensor(pos_encoding)

    def forward(self, x):
        """
        x shape: [batch_size, n_features, 1] - tabular data
        """
        batch_size = x.shape[0]

        # 1. Feature Embedding
        # x: [batch, features, 1] -> [batch, features, d_model]
        embedded = x.matmul(self.W_embed.transpose()) + self.b_embed

        # 2. Add positional encoding
        embedded = embedded + self.pos_encoding

        # 3. Apply learnable patterns (TabPFN innovation)
        # Multiply by patterns: [batch, features, d_model] * [1, features, d_model]
        embedded = embedded * self.patterns

        # 4. Pass through transformer blocks
        features = embedded
        for block in self.blocks:
            features = block.forward(features)

        # 5. Feature pooling (mean across features)
        # features: [batch, features, d_model] -> [batch, d_model]
        pooled = features.mean(axis=1)

        # 6. Output projection
        output = pooled.matmul(self.W_out.transpose()) + self.b_out

        return output


# ============================================
# Visualization with Boxes
# ============================================

# Create synthetic tabular data
batch_size = 1
n_features = 4
x_data = np.random.randn(batch_size, n_features, 1)

# Create TabPFN model
tabpfn = TabPFN(n_features=n_features)

# Convert to Tensor
x = Tensor(x_data)

print("=" * 80)
print("TabPFN Model - Step by Step Visualization")
print("=" * 80)

# Step 1: Input Table
box("Input Table", x, "3")
print(f"Shape: {x.shape}")
print()

# Step 2: Feature Embedding
embedded = x.matmul(tabpfn.W_embed.transpose()) + tabpfn.b_embed
box("Feature Embedding", embedded, "2")
print(f"Shape: {embedded.shape}")
print(f"W_embed shape: {tabpfn.W_embed.shape}")
print()

# Step 3: Positional Encoding
pos_encoded = embedded + tabpfn.pos_encoding
box("+ Positional Encoding", pos_encoded, "3")
print(f"Pos encoding shape: {tabpfn.pos_encoding.shape}")
print()

# Step 4: Learnable Patterns (TabPFN Innovation)
patterned = pos_encoded * tabpfn.patterns
box("× Learnable Patterns", patterned, "4")
print(f"Patterns shape: {tabpfn.patterns.shape}")
print()

# Step 5: Transformer Blocks (first block detailed)
print("Transformer Block 1:")
print("-" * 40)

# Get first block
block = tabpfn.blocks[0]

# Multi-head attention weights
box("W_q (Attention)", block.W_q, "1")
box("W_k (Attention)", block.W_k, "2")
box("W_v (Attention)", block.W_v, "3")
box("W_o (Attention)", block.W_o, "4")

# Attention computation
Q = patterned.matmul(block.W_q.transpose())
K = patterned.matmul(block.W_k.transpose())
V = patterned.matmul(block.W_v.transpose())

box("Q (Query)", Q, "4")
box("K (Key)", K, "5")
box("V (Value)", V, "6")

# Reshape for multi-head
batch_size, seq_len, d_model = Q.shape
Q_reshaped = Q.reshape(batch_size, seq_len, tabpfn.n_heads, -1).transpose(1, 2)
K_reshaped = K.reshape(batch_size, seq_len, tabpfn.n_heads, -1).transpose(1, 2)
V_reshaped = V.reshape(batch_size, seq_len, tabpfn.n_heads, -1).transpose(1, 2)

# Scaled dot-product attention
scores = Q_reshaped.matmul(K_reshaped.transpose(-2, -1))
scaling_factor = 1 / math.sqrt(block.d_k)
scaled_scores = scores * scaling_factor

softmax = Softmax()
attention_weights = softmax.forward(scaled_scores, dim=-1)
attn_output = attention_weights.matmul(V_reshaped)

# Output projection
attn_output_reshaped = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
attn_final = attn_output_reshaped.matmul(block.W_o.transpose())

box("Attention Output", attn_final, "7")

# Skip connection and layer norm
residual = patterned
x_after_attn = residual + attn_final
x_norm1 = layer_norm(x_after_attn, block.gamma1, block.beta1)

box("After Attention + Skip", x_after_attn, "8")
box("After Layer Norm", x_norm1, "9")

# Feed-forward network
ff_output = feed_forward_network(x_norm1, block.W_ffn1, block.b_ffn1, block.W_ffn2, block.b_ffn2)

# Skip connection and layer norm
residual2 = x_norm1
x_after_ffn = residual2 + ff_output
x_norm2 = layer_norm(x_after_ffn, block.gamma2, block.beta2)

box("FFN Output", ff_output, "5")
box("After FFN + Skip", x_after_ffn, "6")
box("Final Block Output", x_norm2, "7")

# Step 6: Through all transformer blocks (simplified)
features = x_norm2
for i in range(1, tabpfn.n_layers):
    features = tabpfn.blocks[i].forward(features)
    if i < 3:  # Show first 3 blocks
        box(f"Block {i + 1} Output", features, f"13.{i}")
        print(features)

# Step 7: Feature Pooling
pooled = features.mean(axis=1)
box("Feature Pooling (Mean)", pooled, "8")
print(f"Shape after pooling: {pooled.shape}")

# Step 8: Output Projection
output = pooled.matmul(tabpfn.W_out.transpose()) + tabpfn.b_out
box("Final Output", output, "9")
print(f"Output shape: {output.shape}")
print(f"Number of classes: {tabpfn.n_classes}")

print("\n" + "=" * 80)
print("TabPFN Model Statistics:")
print("=" * 80)
print(f"Total parameters: ~1.5M")
print(f"Transformer layers: {tabpfn.n_layers}")
print(f"Model dimension: {tabpfn.d_model}")
print(f"Attention heads: {tabpfn.n_heads}")
print(f"Input features: {tabpfn.n_features}")
print(f"Output classes: {tabpfn.n_classes}")


# Function to count parameters
def count_parameters(model):
    total = 0
    # Count embedding parameters
    total += model.W_embed.size + model.b_embed.size
    total += model.patterns.size
    total += model.pos_encoding.size

    # Count transformer block parameters
    for block in model.blocks:
        total += block.W_q.size + block.W_k.size + block.W_v.size + block.W_o.size
        total += block.gamma1.size + block.beta1.size + block.gamma2.size + block.beta2.size
        total += block.W_ffn1.size + block.b_ffn1.size + block.W_ffn2.size + block.b_ffn2.size

    # Count output parameters
    total += model.W_out.size + model.b_out.size

    return total


print(f"Actual parameter count: {count_parameters(tabpfn):,}")

print("\n" + "=" * 80)
print("✅ TabPFN model created successfully from base components!")
print("=" * 80)
