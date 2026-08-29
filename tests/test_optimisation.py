import numpy as np
import pytest
from src.utils.optimisation import gradient_descent

# Test target function: f(x) = sum(x^2), true minimum is at origin (0.0)
def sum_of_squares(x):
    return np.sum(x**2)

@pytest.mark.parametrize("initial_point, learning_rate, n_iterations", [
    (np.array([4.0]), 0.1, 50),                 # 1D input
    (np.array([4.0, 4.0]), 0.1, 50),            # 2D input
    (np.array([3.0, -2.0, 5.0]), 0.1, 50),      # 3D input
])
def test_gradient_descent(initial_point, learning_rate, n_iterations):
    path = gradient_descent(
        sum_of_squares, 
        initial_point=initial_point, 
        learning_rate=learning_rate, 
        n_iterations=n_iterations
    )

    # 1. Verify output trajectory shape: (n_iterations + 1, d)
    assert path.shape == (n_iterations + 1, len(initial_point))

    # 2. Verify starting position matches initial point
    assert np.allclose(path[0], initial_point)

    # 3. Verify final point converged close to the minimum [0, 0, ...]
    assert np.allclose(path[-1], np.zeros_like(initial_point), atol=1e-3)