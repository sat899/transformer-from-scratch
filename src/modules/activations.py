"""
activations.py

Common activation functions used in transformer architectures, including 
ReLU, GELU, and Swish/SiLU. Features raw NumPy routines with explicit manual 
derivatives for backpropagation alongside standard PyTorch functional modules.
"""

import numpy as np

def softmax(scores, subtract_max = True):
    """
    Assuming that scores have dimension [batch_size, seq_len, vocab_size]
    
    """
    # should we do a check here of the dimensions?
    
    if subtract_max:
        scores = scores - np.max(scores, axis=-1, keepdims=True) # subtracting max value before exponentiating to prevent numerical overflow.

    print('calculating the numerator = e^x for each value of x in the input tensor of scores')
    numerator = np.exp(scores) # np.exp() calculates e^x for each value of x in your input array/tensor. Element-wise, shape does not change.
    print(f'the nummerator is {numerator}')
    print('calculating the denominator = summing across the last dimension of the numerator (vocab_size)')
    denominator = np.sum(numerator, axis=-1, keepdims=True) # sums across the last dimension which is the vocab size, so getting a probability distribution over all words in the vocab. Shape is [batch_size, seq_len, 1]
    print(f'the denominator is {denominator}')
    prob = numerator / denominator # same shape as scores
    return prob