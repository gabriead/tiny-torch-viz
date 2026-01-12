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
Based on [TinyTorch](https://mlsysbook.ai/tinytorch/intro.html) and the educational concepts of [AI by Hand](https://www.byhand.ai/), making it perfect for learning ML fundamentals ahead of the curve.

## 🚀 Quick Start

**Try it instantly on Hugging Face Spaces:**
[**Launch TinyTorch Viz**](https://huggingface.co/spaces/gabriead/tiny-torch-viz)


### Or run locally:
```bash
cd /path/to/TinyTorch
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.


---

## ✨ Features

- **Visual Code Execution:** Write TinyTorch code and instantly generate a dynamic visual graph of your tensors and operations. This connects abstract code to concrete visual structures, deepening your understanding of neural network mechanics.
    
- **Build Architectures from Scratch:** Create complete Deep Learning models layer-by-layer, visualizing the flow of data and gradients through every component.
    
- **Scalable Visualization:** Toggle the "Data" view to hide raw tensor values and focus purely on architecture. This allows you to design larger, complex networks without clutter.
    
- **Side-by-Side PDF Learning:** Open research papers or tutorials directly within the app to code alongside the text, enabling immediate implementation and verification of concepts.
    
- **Rich Annotations:** Add notes with full LaTeX support to explain mathematical formulas, document layer logic, or leave reminders directly on your visual graph.
    
- **Save & Share:** Export your entire workspace—including code, visual graph, and notes—as a JSON file to share your educational content or resume work later.
    
- **Retro Mode:** Switch to a distinct retro-styled UI for a focused, nostalgic coding experience.
    

## ⚠️ Important Note on Math Notation

**I changed the order of matrix multiplication from $x \cdot W$ to $W \cdot x$.**

This follows the [AI by Hand](https://www.byhand.ai/) teaching methodology. While standard libraries often use `input @ weights`, this tool visualizes weights first to align with standard mathematical notation found in linear algebra textbooks.

---

## 📚 Core API Reference

Please look at the great work done by [TinyTorch](https://mlsysbook.ai/tinytorch/intro.html) for a complete API reference. Below are the currently enabled features for visualization.

### 📦 Tensor

The fundamental data structure.

Python

```
import numpy as np

# Create tensors
a = Tensor([1, 2, 3, 4])
b = Tensor(np.random.randn(3, 4))
```

#### Arithmetic Operations

Python

```
c = a + b       # Addition
d = a - b       # Subtraction
e = a * b       # Multiplication
f = a / b       # Division
```

#### Matrix Operations

Python

```
a = Tensor([[1, 2, 3], [4, 5, 6]])      # Shape: (2, 3)
b = Tensor([[1, 2], [3, 4], [5, 6]])    # Shape: (3, 2)

# Matrix multiplication
c = a @ b              # Using @ operator
c = a.matmul(b)        # Using method

# Transpose
d = a.transpose()      # Swap last two dimensions
```

#### Reduction Operations

Python

```
total = a.sum()                 # Sum all elements
col_sums = a.sum(axis=0)        # Sum each column
avg = a.mean()                  # Mean of all elements
maximum = a.max()               # Max of all elements
```

### ⚡ Activations

Non-linear functions that enable neural networks to learn complex patterns.

- `ReLU()`
    
- `Sigmoid()`
    
- `Tanh()`
    
- `GELU()`
    
- `Softmax()`
    

### 🧱 Layers

Building blocks for neural network architectures.

#### Linear Layer

Fully connected layer: $y = Wx + b$ (Note the order!)

Python

```
# Create layer: 8 input features → 4 output features
linear = Linear(8, 4)
y = linear(x)
```

#### Dropout

Regularization layer.

Python

```
dropout = Dropout(p=0.5)
dropout.train() # Drop values
y = dropout(x)
```

#### Sequential

Container for stacking layers.

Python

```
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10),
    Softmax()
)
```

### 📉 Loss Functions

- `MSELoss()`: For regression tasks.
    
- `CrossEntropyLoss()`: For multi-class classification.
    

### 🎨 Visualization Commands

The visualizer provides special commands to organize the UI.

Python

```
# Group tensors into a colored box
box("Layer 1", [x, weights, output], "2")
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

Python

```
import numpy as np

# 1. Create Data (Note shapes: Batch=4, Features=8)
X = Tensor(np.random.randn(4, 8))
y = Tensor([0, 1, 2, 1])

# 2. Define Network
layer1 = Linear(8, 16)
relu1 = ReLU()
layer2 = Linear(16, 8)
relu2 = ReLU()
layer3 = Linear(8, 3)
softmax = Softmax()
loss_fn = CrossEntropyLoss()

# 3. Forward Pass
z1 = layer1(X)
a1 = relu1(z1)
z2 = layer2(a1)
a2 = relu2(z2)
logits = layer3(a2)
predictions = softmax(logits)

# 4. Compute Loss
loss = loss_fn(logits, y)

# 5. Visualize
box("Input", X, "1")
box("Layer 1", [z1, a1], "2")
box("Output", [logits, predictions, loss], "4")
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
    └── core/              # Core ML library implementation
```

## License

MIT License - Built for education and understanding.