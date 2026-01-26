import numpy

# TabPFN toy example

# training data

X_train = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
Y_train = Tensor([1, 0])
X_test = Tensor([[9, 10, 11, 12]])
Y_test = Tensor([0])

box("X_train", [X_train, Y_train, X_test], "1")

# Feature Encoder - Feature Embeddings
W_enc = Tensor([[1, 0.5], [0.5, 1], [0.3, 0.7], [0.7, 0.3]])
W_enc_transpose = W_enc.transpose()
b_enc = Tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])

box("Feature Encoder", W_enc_transpose, "2")

# Label Encoder - Label Embeddings
W_y = Tensor([[1], [-1], [0], [0]]).reshape(1, 4)
b_y = Tensor([0, 0, 0, 0])


def label_embeddings(y_train):
    lbl_embds = np.zeros((3, 4))
    for (idx, row) in enumerate(y_train):
        res = row.data * W_y.data
        lbl_embds[idx] = res

    return Tensor(lbl_embds)


y_stacked = Tensor(np.hstack((Y_train.data, Y_test.data)))
label_embeds = label_embeddings(y_stacked)

box("label Encoder", [W_y, label_embeds], "3")

# Step 1: Combine Training and Test Samples
X_combined = X_combined = Tensor(np.vstack([X_train.data, X_test.data]))
box("Training and Test Samples", X_combined, "4")


# Step 1: Tokenization - Group Features

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
            if col == 0:
                X_encoded[idx][0] = group_matmul.data
                col = 1
            else:
                X_encoded[idx][1] = group_matmul.data
                col = 0
        idx += 1
    X_encoded_tensor = Tensor(X_encoded)
    return X_encoded_tensor


X_encoded = group(X_combined)
box("X_encoded", [X_encoded], "4")

# Step 3: Add Thinking Tokens
Thinking_Tokens = Tensor([[[0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]]])

# box("Thinking Tokens", Thinking_Tokens, "4")

# Computing full model input


labels_reshaped = label_embeds.data.reshape(3, 1, 4)
data_rows = np.concatenate([X_encoded.data, labels_reshaped], axis=1)
E_numpy = np.concatenate([Thinking_Tokens.data, data_rows], axis=0)
E = Tensor(E_numpy)
# print(E)

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

scaling_factor = np.sqrt(4)


def compute_attn_labels(labels):
    q


labels = [E[1][2], E[2][2], E[2][2]]
compute_attn_labels(labels)