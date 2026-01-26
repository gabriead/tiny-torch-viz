import numpy as np
import math
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import Softmax, GELU
from tinytorch.core.layers import Dropout

# -----------------------------
# Minimal numpy glue
# -----------------------------
def _np(t: Tensor):
    # adjust if your Tensor uses a different attribute
    return t.data

def concat(tensors, axis):
    return Tensor(np.concatenate([_np(t) for t in tensors], axis=axis))

def repeat_batch(t: Tensor, B: int):
    arr = _np(t)
    if arr.shape[0] == B:
        return t
    return Tensor(np.repeat(arr, B, axis=0))

# -----------------------------
# Your base attention primitives
# -----------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q.matmul(K.transpose(-2, -1))
    scaled_scores = scores * (1.0 / math.sqrt(d_k))

    if mask is not None:
        # mask==1 => forbidden
        scaled_scores = scaled_scores + (mask * -1e9)

    softmax = Softmax()
    A = softmax.forward(scaled_scores, dim=-1)
    out = A.matmul(V)
    return out, A

def multi_head_attention(x, W_q, W_k, W_v, W_o, n_heads, mask=None):
    B, S, D = x.shape
    d_k = D // n_heads

    Q = x.matmul(W_q.transpose())
    K = x.matmul(W_k.transpose())
    V = x.matmul(W_v.transpose())

    Q = Q.reshape(B, S, n_heads, d_k).transpose(1, 2)
    K = K.reshape(B, S, n_heads, d_k).transpose(1, 2)
    V = V.reshape(B, S, n_heads, d_k).transpose(1, 2)

    out, _ = scaled_dot_product_attention(Q, K, V, mask)
    out = out.transpose(1, 2).reshape(B, S, D)
    out = out.matmul(W_o.transpose())
    return out

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) * (x - mean)).mean(axis=-1, keepdims=True)
    std = (var + eps).sqrt()
    return ((x - mean) / std) * gamma + beta

def feed_forward_network(x, W1, b1, W2, b2):
    h = x.matmul(W1.transpose()) + b1
    gelu = GELU()
    h = gelu.forward(h)
    y = h.matmul(W2.transpose()) + b2
    return y

# -----------------------------
# Feature grouping (size = 3)
# -----------------------------
def group_features(X, group_size=3):
    """
    X: [B, R, F, 1]
    returns Xg: [B, R, G, group_size] where G=F//group_size
    """
    arr = _np(X)
    B, R, F, one = arr.shape
    assert one == 1
    assert F % group_size == 0
    G = F // group_size
    arr = arr.reshape(B, R, G, group_size)
    return Tensor(arr)

def group_linear_embed(Xg, W, b):
    """
    Xg: [B, R, G, I]  (I = group_size)
    W:  [D, I]
    b:  [D]
    returns: [B, R, G, D]
    """
    arr = _np(Xg)
    B, R, G, I = arr.shape
    # reshape to [B*R*G, 1, I] so we can matmul with W^T => [B*R*G, 1, D]
    x = Tensor(arr.reshape(B * R * G, 1, I))
    y = x.matmul(W.transpose()) + b
    return Tensor(_np(y).reshape(B, R, G, W.shape[0]))

# -----------------------------
# Masks
# -----------------------------
def make_row_attention_mask(n_think, n_train, n_test, forbid_test_to_self=False):
    """
    mask: [1,1,R,R], mask==1 => forbidden
    R = n_think + n_train + n_test
    """
    R = n_think + n_train + n_test
    m = np.zeros((R, R), dtype=np.float32)

    th0 = 0
    tr0 = n_think
    te0 = n_think + n_train

    # train rows cannot attend to test rows
    if n_test > 0:
        m[tr0:te0, te0:R] = 1.0

    # test rows cannot attend to other test rows
    for i in range(te0, R):
        m[i, te0:R] = 1.0
        m[i, i] = 0.0

    if forbid_test_to_self:
        for i in range(te0, R):
            m[i, i] = 1.0

    return Tensor(m.reshape(1, 1, R, R))

def make_column_attention_mask(C, y_index, feature_only_for_features=True):
    """
    Simple column mask for toy/debug:
    - feature columns (0..y_index-1) attend only to themselves if feature_only_for_features=True
    - y column can attend to all columns (default)
    mask: [1,1,C,C]
    """
    m = np.zeros((C, C), dtype=np.float32)
    if feature_only_for_features:
        for i in range(y_index):
            for j in range(C):
                if j != i:
                    m[i, j] = 1.0
    # y_index row left as zeros => can attend to all
    return Tensor(m.reshape(1, 1, C, C))

# -----------------------------
# Alternating block (columns then rows)
# -----------------------------
class TabPFN25AlternatingBlock:
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads

        # Column-attn weights
        self.Wq_c = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.Wk_c = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.Wv_c = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.Wo_c = Tensor(np.random.randn(d_model, d_model) * 0.02)

        # Row-attn weights
        self.Wq_r = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.Wk_r = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.Wv_r = Tensor(np.random.randn(d_model, d_model) * 0.02)
        self.Wo_r = Tensor(np.random.randn(d_model, d_model) * 0.02)

        # Norm params
        self.gamma_c = Tensor(np.ones((d_model,)))
        self.beta_c  = Tensor(np.zeros((d_model,)))
        self.gamma_r = Tensor(np.ones((d_model,)))
        self.beta_r  = Tensor(np.zeros((d_model,)))
        self.gamma_f = Tensor(np.ones((d_model,)))
        self.beta_f  = Tensor(np.zeros((d_model,)))

        # FFN
        self.W1 = Tensor(np.random.randn(d_model * 4, d_model) * 0.02)
        self.b1 = Tensor(np.zeros((d_model * 4,)))
        self.W2 = Tensor(np.random.randn(d_model, d_model * 4) * 0.02)
        self.b2 = Tensor(np.zeros((d_model,)))

        self.dropout = Dropout(dropout)

    def forward(self, E, row_mask=None, col_mask=None, training=True):
        """
        E: [B, R, C, D]
        """
        B, R, C, D = E.shape

        # ---- Column attention (within each row) ----
        x = E.reshape(B * R, C, D)                       # [B*R, C, D]
        attn = multi_head_attention(
            x, self.Wq_c, self.Wk_c, self.Wv_c, self.Wo_c,
            self.n_heads, mask=col_mask
        )
        attn = self.dropout.forward(attn, training=training)
        x = layer_norm(x + attn, self.gamma_c, self.beta_c)
        E = x.reshape(B, R, C, D)

        # ---- Row attention (within each column) ----
        x = E.transpose(0, 2, 1, 3).reshape(B * C, R, D)  # [B*C, R, D]
        attn = multi_head_attention(
            x, self.Wq_r, self.Wk_r, self.Wv_r, self.Wo_r,
            self.n_heads, mask=row_mask
        )
        attn = self.dropout.forward(attn, training=training)
        x = layer_norm(x + attn, self.gamma_r, self.beta_r)
        E = x.reshape(B, C, R, D).transpose(0, 2, 1, 3)   # [B,R,C,D]

        # ---- FFN (cell-wise) ----
        ff = feed_forward_network(E, self.W1, self.b1, self.W2, self.b2)
        ff = self.dropout.forward(ff, training=training)
        E = layer_norm(E + ff, self.gamma_f, self.beta_f)

        return E

# -----------------------------
# Full TabPFN-2.5-like tiny model
# -----------------------------
class TabPFN25TinyTorch:
    def __init__(self,
                 n_features,
                 group_size=3,
                 d_model=256,
                 n_heads=8,
                 n_layers=12,
                 n_classes=2,
                 dropout=0.1,
                 n_thinking_rows=64):

        assert n_features % group_size == 0
        self.n_features = n_features
        self.group_size = group_size
        self.n_groups = n_features // group_size
        self.n_classes = n_classes
        self.n_think = n_thinking_rows

        # Encoders
        self.W_x = Tensor(np.random.randn(d_model, group_size) * 0.02)
        self.b_x = Tensor(np.zeros((d_model,)))

        self.W_y = Tensor(np.random.randn(d_model, 1) * 0.02)
        self.b_y = Tensor(np.zeros((d_model,)))

        # Learned column embeddings for C = n_groups + 1
        C = self.n_groups + 1
        self.col_embed = Tensor(np.random.randn(1, 1, C, d_model) * 0.02)

        # Learned thinking rows in embedding space
        if self.n_think > 0:
            self.think_rows = Tensor(np.random.randn(1, self.n_think, C, d_model) * 0.02)
        else:
            self.think_rows = None

        self.blocks = [TabPFN25AlternatingBlock(d_model, n_heads, dropout) for _ in range(n_layers)]

        # Readout from target column
        self.W_out = Tensor(np.random.randn(n_classes, d_model) * 0.02)
        self.b_out = Tensor(np.zeros((n_classes,)))

    def forward(self, X_train, y_train, X_test,
                training=True,
                col_mask=None,
                forbid_test_to_self=False):
        """
        X_train: [B, Rtr, F, 1]
        y_train: [B, Rtr, 1]  (or [B,Rtr])
        X_test : [B, Rte, F, 1]
        returns logits: [B, Rte, n_classes]
        """
        if len(y_train.shape) == 2:
            y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

        B, Rtr, F, _ = X_train.shape
        Rte = X_test.shape[1]
        G = self.n_groups
        C = G + 1
        y_col = G

        # y_test placeholder: mean of y_train
        y_mean = y_train.mean(axis=1, keepdims=True)               # [B,1,1]
        y_test = y_mean * Tensor(np.ones((1, Rte, 1), dtype=np.float32))

        # Stack rows
        X_all = concat([X_train, X_test], axis=1)                   # [B, R, F, 1]
        y_all = concat([y_train, y_test], axis=1)                   # [B, R, 1]
        R = Rtr + Rte

        # Feature grouping & embedding
        Xg = group_features(X_all, self.group_size)                 # [B, R, G, group_size]
        E_x = group_linear_embed(Xg, self.W_x, self.b_x)            # [B, R, G, D]

        # y embedding into last column
        y_all = y_all.reshape(B, R, 1, 1)                           # [B,R,1,1]
        E_y = y_all.matmul(self.W_y.transpose()) + self.b_y         # [B,R,1,D]

        # Table: [B,R,C,D]
        E = concat([E_x, E_y], axis=2)
        E = E + self.col_embed

        # Thinking rows
        if self.think_rows is not None:
            think = repeat_batch(self.think_rows, B)
            E = concat([think, E], axis=1)                          # [B, T+R, C, D]

        # Row mask
        row_mask = make_row_attention_mask(self.n_think, Rtr, Rte, forbid_test_to_self=forbid_test_to_self)

        # Blocks
        for blk in self.blocks:
            E = blk.forward(E, row_mask=row_mask, col_mask=col_mask, training=training)

        # Readout: test rows target column
        te0 = self.n_think + Rtr
        te1 = self.n_think + Rtr + Rte
        Z = E[:, te0:te1, y_col, :]                                 # [B,Rte,D]
        logits = Z.matmul(self.W_out.transpose()) + self.b_out       # [B,Rte,n_classes]
        return logits

    def predict_with_permutation_ensemble(self, X_train, y_train, X_test, perms):
        """
        perms: list of permutations of feature indices (length = F)
        returns mean logits over perms: [B,Rte,n_classes]
        """
        logits_sum = None
        for p in perms:
            p = np.array(p, dtype=np.int64)
            Xt = Tensor(_np(X_train)[:, :, p, :])
            Xq = Tensor(_np(X_test)[:, :, p, :])
            logits = self.forward(Xt, y_train, Xq, training=False)
            logits_sum = logits if logits_sum is None else (logits_sum + logits)
        return logits_sum * (1.0 / len(perms))
