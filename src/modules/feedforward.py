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
    maps the input features to output classes, optionally adding a bias term
    y = xw + b
    can use either np.matmul() or numpy's built in matrix multiplication operator here (@), they are equivalent

    input has shape [n_inputs, input_size]
    weights has shape [input_size, output_size]
    bias has shape [1, output_size]
    output has shape [n_inputs, output_size]
    labels_sparse has shape [n_inputs]
    labels_one_hot has shape [n_inputs, output_size]

    """
    # note they must be multiplued in this order so that the inner dimensions match up
    logits = np.matmul(input, weights)
    #output = input @ weights

    if bias is not None:
        logits += bias  # NumPy broadcasting handles matching dimensions automatically

    return logits