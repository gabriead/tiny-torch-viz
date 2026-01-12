---
title: TinyTorch Viz
emoji: 🔥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---


# TinyTorch Viz 🔥

A lightweight, educational framework built to visualize your deep learning architectures and foster a deep mathematical intuition.
Based on TinyTorch [TinyTorch](https://mlsysbook.ai/tinytorch/intro.html) and the educational concepts of [AI by Hand](https://www.byhand.ai/). 
making it perfect for learning ML fundamentals ahead of the curve.

## What can you do with it?
You can use the code editor to code deep learning architectures and directly see the visualiuzation in order develop a deep mathematical intuition.

## Important t know
I changed the order of the matrix multplication from x*w to w*x folowing AibyHand teaching methodoloy

## Features

- **Interactive Visualization**: Build and visualize your deep learning architecture in the browser
- **Educational Design**: Clear, readable code focused on understanding
- **Complete ML Stack**: Tensors, layers, activations, losses, optimizers, and more
- **NumPy Backend**: Pure Python/NumPy implementation
---

## Quick Start

### Running the Visualization Server

```bash
cd /path/to/TinyTorch
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser to access the interactive visualization.

---

## Core API Reference

### 📦 Tensor

The fundamental data structure for all operations.

```python
import numpy as np

# Create tensors
a = Tensor([1, 2, 3, 4])
b = Tensor(np.random.randn(3, 4))
c = Tensor([[1, 2], [3, 4]])

```

#### Arithmetic Operations

```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[5, 6], [7, 8]])

# Element-wise operations
c = a + b       # Addition
d = a - b       # Subtraction
e = a * b       # Multiplication
f = a / b       # Division
```

#### Matrix Operations

```python
a = Tensor([[1, 2, 3], [4, 5, 6]])      # Shape: (2, 3)
b = Tensor([[1, 2], [3, 4], [5, 6]])    # Shape: (3, 2)

# Matrix multiplication
c = a @ b              # Using @ operator
c = a.matmul(b)        # Using method

# Transpose
d = a.transpose()      # Swap last two dimensions
```

#### Shape Operations

```python
a = Tensor([1, 2, 3, 4, 5, 6])

# Reshape
b = a.reshape(2, 3)    # Shape: (2, 3)
c = a.reshape(3, 2)    # Shape: (3, 2)
d = a.reshape(-1, 2)   # Auto-infer: (3, 2)
```

#### Reduction Operations

```python
a = Tensor([[1, 2, 3], [4, 5, 6]])

# Sum
total = a.sum()                 # Sum all elements → scalar
row_sums = a.sum(axis=1)        # Sum each row → [6, 15]
col_sums = a.sum(axis=0)        # Sum each column → [5, 7, 9]

# Mean
avg = a.mean()                  # Mean of all elements
row_means = a.mean(axis=1)      # Mean of each row

# Max
maximum = a.max()               # Max of all elements
row_max = a.max(axis=1)         # Max of each row
```

---

### ⚡ Activations

Non-linear functions that enable neural networks to learn complex patterns.

#### ReLU (Rectified Linear Unit)

```python
relu = ReLU()
x = Tensor([-2, -1, 0, 1, 2])
y = relu(x)  # [0, 0, 0, 1, 2]
```

#### Sigmoid

```python
sigmoid = Sigmoid()
x = Tensor([-2, 0, 2])
y = sigmoid(x)  # [0.119, 0.5, 0.881]
```

#### Tanh

```python
tanh = Tanh()
x = Tensor([-2, 0, 2])
y = tanh(x)  # [-0.964, 0, 0.964]
```

#### GELU (Gaussian Error Linear Unit)

```python
gelu = GELU()
x = Tensor([-2, 0, 2])
y = gelu(x)  # Modern activation used in transformers
```

#### Softmax

```python
softmax = Softmax()
logits = Tensor([[1.0, 2.0, 3.0]])
probs = softmax(logits)  # [[0.09, 0.24, 0.67]] - sums to 1
```

---

### 🧱 Layers

Building blocks for neural network architectures.

#### Linear Layer

Fully connected layer: `y = xW + b`

```python
# Create layer: 8 input features → 4 output features
linear = Linear(8, 4)

# Forward pass
x = Tensor(np.random.randn(32, 8))  # Batch of 32 samples
y = linear(x)                        # Output shape: (32, 4)

```

#### Dropout

Regularization layer that randomly zeros elements during training.

```python
dropout = Dropout(p=0.5)  # 50% dropout rate

x = Tensor(np.random.randn(32, 64))

# Training mode (drops values)
dropout.train()
y_train = dropout(x)

# Evaluation mode (no dropout)
dropout.eval()
y_eval = dropout(x)
```

#### Sequential

Container for stacking layers.

```python
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
    Softmax()
)

x = Tensor(np.random.randn(32, 784))
y = model(x)  # Shape: (32, 10)
```

---

### 📉 Loss Functions

Measure how wrong predictions are.

#### MSELoss (Mean Squared Error)

For regression tasks.

```python
loss_fn = MSELoss()

predictions = Tensor([[1.0, 2.0], [3.0, 4.0]])
targets = Tensor([[1.5, 2.5], [2.5, 3.5]])

loss = loss_fn(predictions, targets)
print(loss.data)  # Scalar loss value
```

#### CrossEntropyLoss

For multi-class classification. Takes logits (raw scores) and class indices.

```python
loss_fn = CrossEntropyLoss()

# Logits: batch of 4 samples, 3 classes
logits = Tensor(np.random.randn(4, 3))

# Targets: class indices (0, 1, or 2)
targets = Tensor([0, 1, 2, 1])

loss = loss_fn(logits, targets)
print(loss.data)
```

---

### 🎯 Optimizers

Update model parameters to minimize loss.

#### SGD (Stochastic Gradient Descent)

```python
model = Linear(10, 5)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training step
loss = compute_loss(model, x, y)
loss.backward()          # Compute gradients
optimizer.step()         # Update parameters
optimizer.zero_grad()    # Clear gradients
```

#### Adam

Adaptive learning rate optimizer.

```python
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

#### AdamW

Adam with decoupled weight decay.

```python
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

### 📊 DataLoader

Batch and iterate over datasets.

#### TensorDataset

```python
# Create dataset from tensors
X = Tensor(np.random.randn(1000, 784))
y = Tensor(np.random.randint(0, 10, 1000))

dataset = TensorDataset(X, y)
print(len(dataset))  # 1000
```

#### DataLoader

```python
loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True
)

for batch_x, batch_y in loader:
    # batch_x shape: (32, 784)
    # batch_y shape: (32,)
    predictions = model(batch_x)
    loss = loss_fn(predictions, batch_y)
```

#### Data Augmentation

```python
transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=(28, 28), padding=4)
])

# Apply during dataset creation
augmented_x = transform(x)
```

---

### 🔢 Embeddings

Convert discrete tokens to continuous vectors.

#### Embedding Layer

```python
# Vocabulary of 1000 tokens → 256-dimensional embeddings
embedding = Embedding(num_embeddings=1000, embedding_dim=256)

# Token indices
tokens = Tensor([5, 23, 100, 42])

# Get embeddings
vectors = embedding(tokens)  # Shape: (4, 256)
```

#### Positional Encoding

Add position information to embeddings (for transformers).

```python
pos_enc = PositionalEncoding(d_model=256, max_len=512)

# Add positional encodings to embeddings
x = embedding(tokens)
x_with_pos = pos_enc(x)
```

#### Combined Embedding Layer

```python
embed_layer = EmbeddingLayer(
    vocab_size=1000,
    embedding_dim=256,
    max_seq_len=512
)

tokens = Tensor([[1, 2, 3, 4, 5]])  # Sequence of token IDs
embeddings = embed_layer(tokens)    # Shape: (1, 5, 256)
```

---

### 📝 Tokenization

Convert text to token sequences.

#### Character Tokenizer

```python
tokenizer = CharTokenizer()

# Build vocabulary
tokenizer.fit(["hello world", "machine learning"])

# Encode text to tokens
tokens = tokenizer.encode("hello")  # [2, 3, 4, 4, 5]

# Decode tokens back to text
text = tokenizer.decode(tokens)     # "hello"
```

#### BPE Tokenizer

Byte Pair Encoding for subword tokenization.

```python
tokenizer = BPETokenizer(vocab_size=1000)

# Train on corpus
tokenizer.fit(["The quick brown fox", "jumps over the lazy dog"])

# Encode
tokens = tokenizer.encode("The quick fox")

# Decode
text = tokenizer.decode(tokens)
```

---

### 🏋️ Training Utilities

#### Learning Rate Scheduler

```python
scheduler = CosineSchedule(
    optimizer,
    warmup_steps=100,
    total_steps=1000,
    min_lr=1e-6
)

for step in range(1000):
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate
```

#### Gradient Clipping

```python
# Prevent exploding gradients
clip_grad_norm(model.parameters(), max_norm=1.0)
```

#### Trainer

High-level training loop.

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    scheduler=scheduler
)

trainer.fit(
    train_loader,
    val_loader=val_loader,
    epochs=10
)
```

---

### 🔄 Autograd

Automatic differentiation for backpropagation.

```python
# Enable autograd mode
enable_autograd()

# Forward pass creates computation graph
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = x * 2
z = y.sum()

# Backward pass computes gradients
z.backward()
print(x.grad)  # Gradients of z with respect to x
```

---

## 🎨 Visualization Commands

The TinyTorch visualizer provides special commands for the UI.

### box()

Group tensors together in a labeled box.

```python
# Single tensor
box("Input", input_tensor, "1")

# Multiple tensors in one box
box("Layer 1", [x, weights, output], "2")

# Nested boxes
box("Encoder", [layer1_output, layer2_output], "3", parent="Model")
```

**Color Schemes:**
- `"1"` - Default (dark)
- `"2"` - Green (ReLU/activation)
- `"3"` - Blue (Linear layers)
- `"4"` - Purple (Softmax)
- `"5"` - Orange (Sigmoid)
- `"6"` - Red (Dropout/Loss)

---

## 🧪 Complete Example: 3-Layer Neural Network

```python
import numpy as np

# 1. Create Data
batch_size = 4
input_features = 8
num_classes = 3

X = Tensor(np.random.randn(batch_size, input_features))
y = Tensor([0, 1, 2, 1])  # Class indices

# 2. Define Network
layer1 = Linear(input_features, 16)
relu1 = ReLU()
layer2 = Linear(16, 8)
relu2 = ReLU()
layer3 = Linear(8, num_classes)
softmax = Softmax()

# 3. Loss Function
loss_fn = CrossEntropyLoss()

# 4. Forward Pass
z1 = layer1(X)
a1 = relu1(z1)
z2 = layer2(a1)
a2 = relu2(z2)
logits = layer3(a2)
predictions = softmax(logits)

# 5. Compute Loss
loss = loss_fn(logits, y)

# 6. Visualize with Boxes
box("Input", X, "1")
box("Layer 1: Linear + ReLU", [X, z1, a1], "2")
box("Layer 2: Linear + ReLU", [a1, z2, a2], "3")
box("Output: Linear + Softmax", [a2, logits, predictions], "4")
box("Loss", [predictions, y, loss], "6")

print(f"Input shape: {X.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Loss: {loss.data:.4f}")
```

---

## 📁 Project Structure

```
TinyTorch/
├── app.py                 # FastAPI server for visualization
├── tracer.py              # Tensor operation tracing
├── instrumentation.py     # Hooks for tracing
├── static/
│   └── index.html         # Visualization frontend
└── tinytorch/
    └── core/
        ├── tensor.py      # Tensor class
        ├── activations.py # Activation functions
        ├── layers.py      # Neural network layers
        ├── losses.py      # Loss functions
        ├── optimizers.py  # Optimizers (SGD, Adam)
        ├── dataloader.py  # Data loading utilities
        ├── embeddings.py  # Embedding layers
        ├── tokenization.py# Text tokenizers
        ├── training.py    # Training utilities
        └── autograd.py    # Automatic differentiation
```

---

## License

MIT License - Built for education and understanding.
