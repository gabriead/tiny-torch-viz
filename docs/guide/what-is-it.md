# Introduction to TinyTorch Viz

**TinyTorch Viz** is a lightweight, educational deep learning framework built from scratch. 
It visualizes [TinyTorch](https://mlsysbook.ai/tinytorch/intro.html) and uses educational concepts of [AI by Hand](https://www.byhand.ai/). 
It provides learners  with a unique way of learning TinyTorch with visual support, 
building and visualizing neural networks, making it perfect for learning ML fundamentals.

## Quick Start

To run the visualization server locally:

```bash
cd /path/to/TinyTorch
uv run uvicorn app:app --host 0.0.0.0 --port 8000


::: tip Live DemoClick "Launch App" in the top navigation bar to try the hosted version.:::Core API ReferenceTensorThe fundamental data structure for all operations.::: code-groupPythonimport numpy as np
# Create tensors
a = Tensor([1, 2, 3, 4])
b = Tensor(np.random.randn(3, 4))

print(a.shape)
Python import numpy as np
# Standard NumPy arrays
a = np.array([1, 2, 3, 4])
b = np.random.randn(3, 4)

print(a.shape)
:::LayersBuilding blocks for neural network architectures.Linear LayerFully connected layer: $y = xW + b$Python# Create layer: 8 input features → 4 output features
linear = Linear(8, 4)

# Forward pass
x = Tensor(np.random.randn(32, 8))
y = linear(x)
Visualization CommandsThe visualizer provides special commands to group tensors in the UI.CodeColorIntended Use"1"DarkInputs / Default"2"GreenActivations (ReLU)"3"BlueLinear LayersPython# Group tensors together in a labeled box
box("Layer 1", [x, weights, output], "2")

---

### Final Step
1.  **Move Files:** Ensure your existing `static/` files (index.html, js, css) are moved into `docs/public/app/`.
2.  **Push:** Commit and push these files to GitHub.
3.  **Activate:** Go to Repo Settings -> Pages -> Source: **GitHub Actions**.

Your site will be live at: `https://gabriead.github.io/tiny-torch-viz/`
Your app will be live at: `https://gabriead.github.io/tiny-torch-viz/app/`