import numpy as np

_EXECUTION_LOG = [] # Global log recording forward operations in chronological order

def reset_execution_log():
    """Clears all recorded operations to prepare for a new forward pass."""
    _EXECUTION_LOG.clear()

def record_operation(backward_func, *saved_tensors):
    """
    Appends an executed forward operation and its cached tensors to the log.

    Parameters:
        backward_func (callable): The derivative function to run during the backward pass.
        *saved_tensors (np.ndarray): Tensors from the forward pass required to compute gradients.
    """
    _EXECUTION_LOG.append((backward_func, saved_tensors))

def linear_layer_backward(upstream_grad, inputs, weights, bias=None):
    """
    Computes downstream gradients for a linear_layer i.e. y = xw + b.
    
    Formula: Downstream gradient = upstream gradient * local gradient.

    Transpose various things to make dimensions work.

    Dimensions:
        upstream_grad : [n_inputs, output_size][cite: 1]
        inputs        : [n_inputs, input_size][cite: 1]
        weights       : [input_size, output_size][cite: 1]
        bias          : [1, output_size] or [output_size][cite: 1]
        
    Returns:
        downstream_grad_inputs  : [n_inputs, input_size][cite: 1]
        downstream_grad_weights : [input_size, output_size][cite: 1]
        downstream_grad_bias    : [1, output_size] or None[cite: 1]
    """

    # 1. Gradient with respect to inputs i.e. dl / dx

    # Local gradient, dy / dx = w
    local_grad_wrt_inputs = weights.T # [num_features_out, num_features_in]

    # Downstream gradient, dl / dx = dl / dy * dy / dx
    downstream_grad_inputs = upstream_grad @ local_grad_wrt_inputs # [num_samples, num_features_out] @ [num_features_out, num_features_in] -> [num_samples, num_features_in]

    # 2. Gradient with respect to weights i.e. dl / dw

    # Local gradient, dy / dw = x
    local_grad_wrt_weights = inputs.T # [num_features_in, num_samples]

    # Downstream gradient, dl / dw = dl / dy * dy / dw
    downstream_grad_weights = local_grad_wrt_weights @ upstream_grad # [num_features_in, num_samples] @ [num_samples, num_features_out] -> [num_features_in, num_features_out]

    # 3. Gradient with respect to bias i.e. dl / db
    # Local gradient: dy / db = 1
    # Because the single bias vector is added to every row in the batch during the forward pass, its total gradient is the sum of all upstream errors across the batch

    if bias is not None:
        # Sum upstream gradients across all samples in the batch (axis=0)
        downstream_grad_bias = np.sum(upstream_grad, axis=0, keepdims=True) # [1, num_features_out]
    else:
        downstream_grad_bias = None

    return downstream_grad_inputs, downstream_grad_weights, downstream_grad_bias

def backward(loss_grad):
    """
    Traverses the execution log in reverse to compute gradients for all operations.
    """
    # 1. Reverse the log to work backward from the loss to the inputs
    reversed_log = reversed(_EXECUTION_LOG)

    # 2. Track the incoming upstream gradient for the current operation - starts as the gradient of the loss i.e. loss_grad
    current_upstream_grad = loss_grad

    # 3. Iterate through the network's execution history one layer at a time
    param_grads = []

    for backward_func, saved_tensors in reversed_log:

        # 4. Execute the specific backward math for this operation
        # Pass the incoming error first, then unpack the saved tensors (input, weight, bias)
        downstream_grads = backward_func(current_upstream_grad, *saved_tensors)
        downstream_grad_inputs, downstream_grad_weights, downstream_grad_bias = downstream_grads
        param_grads.append((downstream_grad_weights, downstream_grad_bias))
        current_upstream_grad = downstream_grad_inputs
    
    param_grads.reverse() # put the lust back in forward order
    
    return param_grads