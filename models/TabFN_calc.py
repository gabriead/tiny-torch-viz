import numpy as np
from tinytorch.core.tensor import Tensor


# ============================================
# MINIMAL TABPFN - VERIFIABLE TOY EXAMPLE
# ============================================

class MiniTabPFN:
    """Minimal TabPFN with only 2 features, dimension 2, for manual verification"""

    def __init__(self):
        # Tiny dimensions for verification
        self.n_features = 2
        self.d_model = 2
        self.n_classes = 2

        # Initialize with known values for verification
        # Input embedding
        self.W_embed = Tensor(np.array([[0.5, -0.3], [0.2, 0.8]]).T)  # (2, 2)
        self.b_embed = Tensor(np.array([0.1, 0.2]))

        # Learnable patterns
        self.patterns = Tensor(np.array([[[1.0, 0.5], [0.5, 1.0]]]))

        # Positional encoding (simplified)
        self.pos_encoding = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]]))

        # Single attention head weights (for simplicity)
        self.W_q = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.W_k = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.W_v = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.W_o = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))

        # Layer norm parameters
        self.gamma1 = Tensor(np.array([1.0, 1.0]))
        self.beta1 = Tensor(np.array([0.0, 0.0]))
        self.gamma2 = Tensor(np.array([1.0, 1.0]))
        self.beta2 = Tensor(np.array([0.0, 0.0]))

        # Feed-forward weights (tiny expansion)
        self.W_ffn1 = Tensor(np.array([[0.5, 0.3], [0.2, 0.4], [0.1, 0.2], [0.3, 0.5]]))  # (4, 2)
        self.b_ffn1 = Tensor(np.array([0.1, 0.2, 0.3, 0.4]))
        self.W_ffn2 = Tensor(np.array([[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]]))  # (2, 4)
        self.b_ffn2 = Tensor(np.array([0.1, 0.2]))

        # Output projection
        self.W_out = Tensor(np.array([[1.0, 0.5], [0.5, 1.0]]))
        self.b_out = Tensor(np.array([0.1, 0.2]))

    def layer_norm(self, x, gamma, beta, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) * (x - mean)).mean(axis=-1, keepdims=True)
        std = (var + eps).sqrt()
        normalized = (x - mean) / std
        return normalized * gamma + beta

    def gelu(self, x):
        # Approximate GELU for manual calculation
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def feed_forward(self, x):
        # First linear layer
        hidden = x.matmul(self.W_ffn1.transpose()) + self.b_ffn1

        # GELU activation (simplified)
        hidden_data = np.array(hidden.data)
        hidden_gelu = self.gelu(hidden_data)
        hidden = Tensor(hidden_gelu)

        # Second linear layer
        output = hidden.matmul(self.W_ffn2.transpose()) + self.b_ffn2
        return output

    def attention(self, x):
        # Simple single-head attention
        Q = x.matmul(self.W_q.transpose())
        K = x.matmul(self.W_k.transpose())
        V = x.matmul(self.W_v.transpose())

        # Attention scores
        scores = Q.matmul(K.transpose(-2, -1))
        scaled_scores = scores * (1.0 / np.sqrt(self.d_model))

        # Softmax
        exp_scores = np.exp(scaled_scores.data)
        softmax_scores = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Apply attention
        attention_output = Tensor(softmax_scores).matmul(V)

        # Output projection
        output = attention_output.matmul(self.W_o.transpose())
        return output

    def forward(self, x):
        """Step-by-step forward pass"""
        print("=" * 60)
        print("TOY TABPFN - MANUAL VERIFICATION")
        print("=" * 60)

        # 1. Input
        print(f"\n1. INPUT:\n{x.data}")
        print(f"Shape: {x.shape}")

        # 2. Feature Embedding
        # x: (1, 2, 1), W_embed: (2, 2) -> (1, 2, 2)
        embedded = x.matmul(self.W_embed.transpose()) + self.b_embed
        print(f"\n2. EMBEDDING (x @ W_embed.T + b_embed):")
        print(f"W_embed.T:\n{self.W_embed.transpose().data}")
        print(f"b_embed: {self.b_embed.data}")
        print(f"Result:\n{embedded.data}")

        # 3. Add Positional Encoding
        pos_encoded = embedded + self.pos_encoding
        print(f"\n3. + POSITIONAL ENCODING:")
        print(f"Positional encoding:\n{self.pos_encoding.data}")
        print(f"Result:\n{pos_encoded.data}")

        # 4. Apply Learnable Patterns
        patterned = pos_encoded * self.patterns
        print(f"\n4. × LEARNABLE PATTERNS:")
        print(f"Patterns:\n{self.patterns.data}")
        print(f"Result:\n{patterned.data}")

        # 5. Attention Block
        print(f"\n5. ATTENTION BLOCK:")

        # Self-attention
        attn_output = self.attention(patterned)
        print(f"Attention output:\n{attn_output.data}")

        # Skip connection
        residual1 = patterned
        after_attn = residual1 + attn_output
        print(f"After skip connection:\n{after_attn.data}")

        # Layer norm
        norm1 = self.layer_norm(after_attn, self.gamma1, self.beta1)
        print(f"After layer norm:\n{norm1.data}")

        # 6. Feed-Forward Network
        print(f"\n6. FEED-FORWARD NETWORK:")
        ff_output = self.feed_forward(norm1)
        print(f"FFN output:\n{ff_output.data}")

        # Skip connection
        residual2 = norm1
        after_ffn = residual2 + ff_output
        print(f"After skip connection:\n{after_ffn.data}")

        # Layer norm
        norm2 = self.layer_norm(after_ffn, self.gamma2, self.beta2)
        print(f"After layer norm:\n{norm2.data}")

        # 7. Feature Pooling
        pooled = norm2.mean(axis=1)
        print(f"\n7. FEATURE POOLING (mean across features):")
        print(f"Input shape: {norm2.shape}")
        print(f"Pooled: {pooled.data}")

        # 8. Output Projection
        output = pooled.matmul(self.W_out.transpose()) + self.b_out
        print(f"\n8. OUTPUT PROJECTION:")
        print(f"W_out.T:\n{self.W_out.transpose().data}")
        print(f"b_out: {self.b_out.data}")
        print(f"Final output: {output.data}")

        return output


# ============================================
# MANUAL CALCULATION EXAMPLE
# ============================================

# Create toy data
toy_data = np.array([[[1.0], [2.0]]])  # Batch size 1, 2 features, 1 value each
x_toy = Tensor(toy_data)

print("TOY INPUT DATA:")
print(f"Feature 1: {toy_data[0, 0, 0]:.1f}")
print(f"Feature 2: {toy_data[0, 1, 0]:.1f}")
print()

# Create mini model
mini_tabpfn = MiniTabPFN()

# Run forward pass
output = mini_tabpfn.forward(x_toy)

# ============================================
# MANUAL CALCULATION STEPS
# ============================================

print("\n" + "=" * 60)
print("MANUAL CALCULATION CHECK")
print("=" * 60)

print("\nLet's verify Step 2 (Embedding) manually:")
print("For feature 1 (value = 1.0):")
print("  W_embed.T row 1: [0.5, -0.3]")
print("  b_embed: [0.1, 0.2]")
print("  Result: 1.0 * [0.5, -0.3] + [0.1, 0.2] = [0.6, -0.1]")

print("\nFor feature 2 (value = 2.0):")
print("  W_embed.T row 2: [0.2, 0.8]")
print("  Result: 2.0 * [0.2, 0.8] + [0.1, 0.2] = [0.5, 1.8]")

print("\nEmbedding matrix should be:")
print("  [[0.6, -0.1],")
print("   [0.5, 1.8]]")

print("\nStep 3 (Positional Encoding):")
print("  Positional encoding: [[0.1, 0.2], [0.3, 0.4]]")
print("  Result: [[0.7, 0.1], [0.8, 2.2]]")

print("\nStep 4 (Learnable Patterns):")
print("  Patterns: [[1.0, 0.5], [0.5, 1.0]]")
print("  Element-wise multiply: [[0.7*1.0, 0.1*0.5], [0.8*0.5, 2.2*1.0]]")
print("  Result: [[0.7, 0.05], [0.4, 2.2]]")

"""
3. Differences from Original TabPFNv2:
No causal masking - Original might use it for permutation invariance

Simplified positional encoding - Original might have more sophisticated encoding

No batch normalization - Original might include it

No gradient checkpointing - Not needed for this example

"""