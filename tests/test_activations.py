import numpy as np
import pytest
from src.modules.activations import softmax

@pytest.mark.parametrize("input_array", [
    np.array([1.0, 2.0, 3.0]),                 # 1D array
    np.random.randn(2, 5),                     # 2D array (batch_size, seq_len)
    np.random.randn(2, 4, 8)                   # 3D array (batch_size, seq_len, vocab_size)
])
def test_softmax(input_array):
    out = softmax(input_array)
    
    print("\n--- Softmax Test Info ---")
    print("Input shape:", input_array.shape)
    print("Output shape:", out.shape)
    print("Sum along last axis:", np.sum(out, axis=-1))
    
    # Assert probability sums to 1 along the last axis
    assert np.allclose(np.sum(out, axis=-1), 1.0)