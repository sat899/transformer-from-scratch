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

    input has shape [n_inputs, n_input_features]
    weights has shape [n_input_features, n_output_classes]
    bias has shape [n_output_classes, 1]
    output has shape [n_inputs, n_output_classes]

    """
    # note they must be multiplued in this order so that the inner dimensions match up
    logits = np.matmul(input, weights)
    #output = input @ weights

    if bias is not None:
        logits += bias  # NumPy broadcasting handles matching dimensions automatically

    return logits