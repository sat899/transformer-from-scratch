"""
optimisation.py

Loss functions, gradient computation logic, and optimization algorithms. 
Features raw NumPy routines with explicit analytical gradients (Cross-Entropy, 
SGD, Adam, AdamW) alongside standard PyTorch autograd loss modules and 
optimizer implementations.
"""

import numpy as np

def cross_entropy_loss_one_hot(predictions, labels, eps=1e-15):
    """
    negative log likelihood loss
    log of the predictions (probabilities) multiplied by the labels, summed up, negated, averaged across the batch
    take the log to i) convert multiplication to addition and ii) to penalize probabilities more as they move away from the expected value
    negate so that we turn it into something to be minimized

    computational complexity is O(N x C)
    large memory footprint - have to store a large sparse matrix of size (N,C)
    
    predictions is a probability distribution (num_samples, num_classes)
    labels is a one-hot of the same shape (num_samples, num_classes)
    note: log(0) is -inf so need to avoid 0 predictions, hence add epsilon clipping
    """
    # first clip predictions to prevent log(0) issues
    predictions = np.clip(predictions, eps, 1.0 - eps)

    # calculate loss (log of the predictions multiplied by the labels, summed up across all classes, negated) for each sample
    sample_loss = -np.sum(np.log(predictions) * labels, axis=1)

    # calculate the mean loss across the batch
    mean_loss = np.mean(sample_loss).item()

    return  mean_loss

def cross_entropy_loss_sparse(predictions, labels, eps=1e-15):
    """
    this is a more efficient implementation of cross-entropy loss.
    the one-hot implementation zeros-out all the incorrect predictions, so calculating them is a waste of time
    instead we simply focus in on the index of the correct class, and only compute loss for that position across all samples
    indeed for this reason loss is simple the negative log of the probabilities of the correct class   
    O(N) - only computes the log for the true target therefore there are N logs to compute
    much lower memory footprint, only need to store vector of size (N,)
    
    predictions is a probability distribution (num_samples, num_classes)
    labels is just the integer class indices (num_samples, )
    """
    # first clip predictions to prevent log(0) issues
    predictions = np.clip(predictions, eps, 1.0 - eps)

    # get the number of samples (N)
    N = predictions.shape[0]

    # select only the probabilities of the correct class for each row
    correct_class_probs = predictions[np.arange(N), labels]

    # negative log likelihood of the probabilities of the correct class for each sample
    # note: don't need to sum since we are only looking at the correct class
    # note: don't need to multiply by the labels since we have already selected the correct class, so this would just be multiplying everything by 1
    # just need to take the negative log of the probabilities of the correct class
    sample_loss = -np.log(correct_class_probs)

    # calculate the mean loss across the batch
    mean_loss = np.mean(sample_loss).item()

    return mean_loss