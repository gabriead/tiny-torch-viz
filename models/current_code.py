import numpy

input_data = Tensor([[1.2, 6.1], [8.9, 9.1], [1.0, 2.9], [33.3, 2.2]])
y = Tensor([[3], [3.1], [6.7], [-1]])

box("input_data", [input_data, y])

box("mean", input_data.mean(axis=1, keepdims=True), "4")


def feature_encoder(embedding_size: int, x: Tensor, train_test_split: int):
    # not implemented yet in tiny torch
    # x = x.unsqeeze(-1)
    # Equivalent to unsqueeze(-1)
    x = np.expand_dims(x.data, axis=-1)

    x = x[:, :train_test_split]
    box("train_test_split_unsqueezed", Tensor(x), "4")
    mean = Tensor(x.mean(axis=1, keepdims=True))
    std = Tensor(x.std(axis=1, keepdims=True))

    # equivalent to torch.clip() => np.clip(arr, 0, 10)
    x = Tensor(x)
    norm = (x - mean) / std
    clip = Tensor(np.clip(x.data, -100, 100))

    # we need the linear layer
    linear_layer = Linear(1, 4)
    emb = linear_layer.forward(clip)
    box("encoded_features", [mean, std, norm, clip, emb], "2")


encoded_features = feature_encoder(1, input_data, 3)

