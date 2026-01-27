import numpy

# TabPFN

# training data
X_train = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
Y_train = Tensor([1, 0])
X_test = Tensor([[9, 10, 11, 12]])

box("X_train", [X_train, Y_train, X_test], "1")

# Feature Encoder - Feature Embeddings
W_enc = Tensor([[1, 0.5], [0.5, 1], [0.3, 0.7], [0.7, 0.3]])
W_enc_transpose = W_enc.transpose()
b_enc = Tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])

box("Feature Encoder", W_enc_transpose, "2")

# Feature/group embeddings
E_feat = Tensor([[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]])

box("Group embedding", E_feat, "6")

# Step 1: Combine Training and Test Samples
X_combined = X_combined = Tensor(np.vstack([X_train.data, X_test.data]))
box("Training and Test Samples grouped", X_combined, "4")


# Step 1: Group Features

def group(X):
    groups = X.shape[0] * W_enc.shape[1]
    X_encoded = np.zeros((3, 2, 4))
    # print(X_encoded)
    idx = 0
    col = 0
    for (group_idx, row) in enumerate(X.data):
        rt_ptr = 0
        for rt_ptr in range(0, len(row), 2):
            group_window = Tensor(row[rt_ptr:rt_ptr + 2])
            group_matmul = group_window.matmul(W_enc_transpose) + b_enc[group_idx]
            # group 1
            if col == 0:
                X_encoded[idx][0] = group_matmul.data + E_feat.data[0]
                col = 1
            # group 2
            else:
                X_encoded[idx][1] = group_matmul.data + + E_feat.data[1]
                col = 0
        idx += 1
    X_encoded_tensor = Tensor(X_encoded)
    return X_encoded_tensor


X_encoded = group(X_combined)
box("X_encoded", X_encoded, "4")

# Label Encoder - Label Embeddings
W_y = Tensor([[1, -1, 0, 0], [0, 0, 1, 1]])
b_y = Tensor([0, 0, 0, 0])
y_padded = Tensor([1, 0, np.nan])  # we wan't to mask y_test with nan
y_clean = Tensor([[1, 0, 0], [0, 0, 1]]).reshape(3, 2)
box("y_clean", y_clean, "4")


def label_embeddings(y_train):
    lbl_embds = np.zeros((3, 4))
    for (idx, row) in enumerate(y_train.data):
        res = Tensor((row)).matmul(W_y)
        lbl_embds[idx] = res.data

    return Tensor(lbl_embds)


label_embeds = label_embeddings(y_clean)
# print(label_embeds)

# Step 3: Add Thinking Tokens
Thinking_Tokens = Tensor([
    [[0.01, 0.02, 0.03, 0.04],
     [0.01, 0.02, 0.03, 0.04],
     [0.01, 0.02, 0.03, 0.04]],

    [[0.05, 0.06, 0.07, 0.08],
     [0.05, 0.06, 0.07, 0.08],
     [0.05, 0.06, 0.07, 0.08]]
])
box("Thinking Tokens", Thinking_Tokens, "4")

# Computing full model input

labels_reshaped = label_embeds.data.reshape(3, 1, 4)
data_rows = np.concatenate([X_encoded.data, labels_reshaped], axis=1)
E_numpy = np.concatenate([Thinking_Tokens.data, data_rows], axis=0)
E = Tensor(E_numpy)

# we need to adapt positional embeddings!
# Create row positional embeddings
P_col_pos_embeds = Tensor([[[0.1, 0.1, 0.1, 0.1],
                            [0.2, 0.2, 0.2, 0.2],
                            [0.3, 0.3, 0.3, 0.3]]])

# Add positional embeddings
E = E + P_col_pos_embeds
box("Positional Embedding", E, "9")

# Attention
W_q = Tensor(np.diag([0.1, 0.2, 0.1, 0.2]))
W_k = Tensor(np.diag([0.1, 0.1, 0.1, 0.1]))
W_v = Tensor(np.diag([1, 1, 1, 1]))

box("Attention weights", [W_q, W_k, W_v], "9")
scaling_factor = np.sqrt(4)

# labels = [E[1][2], E[2][2], E[2][2]]
col_att_softmax = Softmax()


def column_attention_inplace(E: Tensor):
    """
    In-place column attention:
      For each item s: X = E[s] has shape (Ttok=3, D=4)
      Does self-attention across the 3 tokens and writes back:
         E[s] <- E[s] + Attn(E[s])
    """
    S, Ttok, D = E.shape
    softmax = Softmax()

    for s in range(S):
        # Snapshot of current item (avoid in-place mixing during compute)
        X = Tensor(E.data[s].copy())  # (3,4)

        Q = X.matmul(W_q.transpose())  # (3,4)
        K = X.matmul(W_k.transpose())  # (3,4)
        V = X.matmul(W_v.transpose())  # (3,4)

        scores = Q.matmul(K.transpose()) / math.sqrt(D)  # (3,3)
        A = softmax.forward(scores, dim=-1)  # (3,3)
        O = A.matmul(V)  # (3,4)

        # In-place residual update of ALL tokens
        E.data[s] = E.data[s] + O.data


column_attention_inplace(E)
box("Updated Logits", E, "5")


def row_attention_inplace(E: Tensor, W_q: Tensor, W_k: Tensor, W_v: Tensor, single_eval_pos: int):
    """
    In-place row attention:
      For each token slot t:
        Q from all S items:      E[:, t, :] -> (S, D)
        K,V from first Klen rows E[:single_eval_pos, t, :] -> (Klen, D)
      Writes:
        E[:, t, :] <- E[:, t, :] + Attn_row(E[:, t, :])
    """
    S, Ttok, D = E.shape
    softmax = Softmax()

    Klen = single_eval_pos
    assert 0 < Klen <= S, "single_eval_pos must be between 1 and S"

    for t in range(Ttok):
        # Snapshot streams (avoid in-place mixing)
        X_all = Tensor(E.data[:, t, :].copy())  # (S, D)
        X_kv = Tensor(E.data[:Klen, t, :].copy())  # (Klen, D)

        Q = X_all.matmul(W_q.transpose())  # (S, D)
        K = X_kv.matmul(W_k.transpose())  # (Klen, D)
        V = X_kv.matmul(W_v.transpose())  # (Klen, D)

        scores = Q.matmul(K.transpose()) / math.sqrt(D)  # (S, Klen)
        A = softmax.forward(scores, dim=-1)  # (S, Klen)
        O = A.matmul(V)  # (S, D)

        # In-place residual update for this token slot
        E.data[:, t, :] = E.data[:, t, :] + O.data
