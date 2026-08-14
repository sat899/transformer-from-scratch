"""
feedforward.py

Position-wise feed-forward networks (FFN) and MLP blocks. Features basic 
two-layer linear projections as well as gated activation variants like 
SwiGLU across NumPy and PyTorch implementations.
"""

import numpy as np

def linear_layer(input, weights, bias=None):
    """
    basic linear transformation of some input matrix by some weights matrix
    y = wx + b
    can use either np.matmul() or numpy's built in matrix multiplication operator here (@), they are equivalent

    taking the vertical approach, treating our input as a column vector
    input has shape [n_input, 1]
    weights has shape [n_output, n_input]
    bias has shape [n_output, 1]
    output has shape [n_output, 1]

    """
    # note they must be multiplued in this order so that the inner dimensions match up
    output = np.matmul(weights, input)
    #output = weights @ input

    if bias is not None:
        output += bias  # NumPy broadcasting handles matching dimensions automatically

    return output