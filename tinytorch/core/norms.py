import numpy as np
from tinytorch.core.tensor import Tensor

class RMSNorm:

    def __init__(self, normalized_shape, eps=1e-6, gamma=1):
        self.normalized_shape = tuple(normalized_shape)
        self.eps = Tensor(eps)
        self.gamma = Tensor(gamma)
        self.weight = Tensor(np.ones(normalized_shape))

        # we need to add elementwise_affine as paramters!


    def forward(self, x:Tensor, dim=-1) -> Tensor:

        """
        Apply rmsnorm along specified dimension.

        Approach:
        1. Get data from x as numpy array (since some of the operations are not available in Tensor)
        2. Square each element of x
        3. Sum the squares of
        4. Add self.epsilon (for numerical stability)
        5. Compute sqrt of the sum of squares
        6. Elementwise divide x by sum of squares
        7. Multiply by self.gamma

        EXAMPLE:
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> rms_norm = RMSNorm()
        >>> norm = rms_norm.forward(x)
        >>> print(norm)
        Tensor([[0.46291    0.92582    1.38873   ]
        [0.78954196 0.98692745 1.1843129 ]])

        """

        squares = x * x
        sum_squares = squares.mean(axis=dim, keepdims=True)
        rms = (self.eps + sum_squares).sqrt()
        y = x / rms
        y = y * self.gamma
        return y * self.weight


    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)


def test_unit_rmsnorm():
    """🧪 Test RMSNorm layer."""
    print("🧪 Unit Test: RMSNorm...")

    # 1. Setup
    # Create a norm layer for a feature dimension of 5
    rms = RMSNorm(normalized_shape=(5,))

    # 2. Test Zero Input
    # 0 / sqrt(0 + eps) * 1 should be 0
    z = Tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    res_z = rms.forward(z)
    assert np.allclose(res_z.data, np.zeros(5)), f"Zero input failed, got {res_z.data}"

    # 3. Test Mathematical Correctness
    # Input: [-10, -1, 0, 1, 10]
    # Squares: [100, 1, 0, 1, 100] -> Sum: 202 -> Mean: 40.4
    # RMS: sqrt(40.4 + 1e-6) ≈ 6.356099
    # Expected: Input / 6.356099
    x = Tensor([-10, -1, 0, 1, 10])
    result = rms.forward(x)

    # Manual calculation for verification
    expected_val = np.array([-10, -1, 0, 1, 10]) / np.sqrt(40.4 + 1e-6)

    # Use allclose with small tolerance for float32 precision
    assert np.allclose(result.data, expected_val, atol=1e-5)

    # Verify specific known values roughly
    # -10 / 6.356... should be approx -1.573
    assert abs(result.data[0] - (-1.57329)) < 1e-3
    assert abs(result.data[2] - 0.0) < 1e-6

    # 4. Test Batch Processing (2D Input)
    # Shape (2, 5) - Batch size 2, Features 5
    batch_x = Tensor([
        [-10, -1, 0, 1, 10],  # Row 1
        [5, 5, 5, 5, 5]  # Row 2 (Constant values)
    ])
    batch_res = rms.forward(batch_x)

    # Row 1 check (Same as before)
    assert np.allclose(batch_res.data[0], expected_val, atol=1e-5)

    # Row 2 check: Constant values should normalize to ~1s (or -1s)
    # Mean(5^2) = 25. RMS = 5. Input/RMS = 1.
    expected_row2 = np.ones(5, dtype=np.float32)
    assert np.allclose(batch_res.data[1], expected_row2, atol=1e-5)

    # 5. Test Shape Preservation
    assert batch_res.shape == (2, 5)

    print("✅ RMSNorm works correctly!")

    #normalized shape does not do anyting at the moment
    rms_norm =  RMSNorm(normalized_shape=(1,5))
    # Test zero
    x = Tensor([0.0])
    result = rms_norm.forward(x)
    assert np.allclose(result.data, [0.0]), f"rms_norm(0) should be 0, got {result.data}"

    x = Tensor([-10, -1, 0, 1, 10])
    result = rms_norm.forward(x)
    print(result)
    expected = np.array([-1.57329, -0.15733, 0., 0.15733, 1.57329], dtype=np.float32)
    assert np.allclose(result.data, expected, atol=1e-4)

    x = Tensor([[-10, -1, 0, 1, 10],[-10, -1, 0, 1, 10]] )
    result = rms_norm.forward(x)
    print(result)
    expected = np.array([[-1.57329,  -0.15733,  0.,          0.15733,  1.57329 ],
                                                 [-1.57329,  -0.15733,  0.,          0.15733,  1.57329]], dtype=np.float32)
    assert np.allclose(result.data, expected, atol=1e-4)


    try:
        incompatible_a = Tensor([[1, 2]])     # 1×2
        incompatible_b = Tensor([[1], [2], [3]])  # 3×1
        incompatible_a.matmul(incompatible_b)  # 1×2 @ 3×1 should fail (2 ≠ 3)
        assert False, "Should have raised ValueError for incompatible shapes"
    except ValueError as e:
        assert "Inner dimensions must match" in str(e)
        assert "2 ≠ 3" in str(e)  # Should show specific dimensions

if __name__ == "__main__":
    test_unit_rmsnorm()