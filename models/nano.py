# Code loads here
# Code loads here
# Code loads here
# Code loads here
# Code loads here
# Code loads here
# Code loads here
import numpy

input_data = input_data = Tensor([
    [
        [1.2, 6.1],
        [8.9, 9.1],
        [1.0, 2.9],
        [33.3, 2.2]
    ]
])
y = Tensor([[[3], [3.1], [6.7], [-1]]])

box("input_data", [input_data, y])


def feature_encoder(embedding_size: int, x: Tensor, train_test_split: int):
    # not implemented yet in tiny torch
    # x = x.unsqeeze(-1)
    # Equivalent to unsqueeze(-1)

    x = np.expand_dims(x.data, axis=-1)

    x = x[:, :train_test_split]
    box("train_test_split_unsqueezed", Tensor(x), "4")
    x = Tensor(x)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)

    # x = Tensor(x)
    norm = (x - mean) / std
    # equivalent to torch.clip() => np.clip(arr, 0, 10)
    clip = Tensor(np.clip(x.data, -100, 100))

    # we need the linear layer
    linear_layer = Linear(1, 4)
    emb = linear_layer(clip)
    box("encoded_features", [mean, std, norm, clip, emb], "2")


encoded_features = feature_encoder(1, input_data, 3)

