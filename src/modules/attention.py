"""
attention.py

Implementations of multi-head attention mechanisms across abstraction levels,
ranging from raw, manual NumPy calculations to PyTorch nn.Modules and 
Hugging Face-compatible wrappers. Includes scaled dot-product logic, 
causal masking, and cross-attention variants.
"""

import numpy as np
from src.modules.linear import linear_layer

def generate_qkv_weight_matrices(embedding_dim: int, head_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate query, key and value weight matrices
    Initialize weight matrices randomly with np.random.randn()
    Add Xavier / Glorot-style scaling
    """
    query_weight_matrix = np.random.randn(embedding_dim, head_dim) * np.sqrt(1.0 / embedding_dim)
    key_weight_matrix = np.random.randn(embedding_dim, head_dim) * np.sqrt(1.0 / embedding_dim)
    value_weight_matrix = np.random.randn(embedding_dim, head_dim) * np.sqrt(1.0 / embedding_dim)

    return query_weight_matrix, key_weight_matrix, value_weight_matrix

def generate_qkv_matrices(input_embedding: np.ndarray, query_weight_matrix: np.ndarray, key_weight_matrix: np.ndarray, value_weight_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the query, key and value matrices from the input embedding and weight matrices
    Input embedding has size [seq_len, embed_dim]
    Weight matrices have size [embed_dim, head_dim]
    Resulting matrices have size [seq_len, head_dim]
    """
    query_matrix = np.matmul(input_embedding, query_weight_matrix)
    key_matrix = np.matmul(input_embedding, key_weight_matrix)
    value_matrix = np.matmul(input_embedding, value_weight_matrix)

    return query_matrix, key_matrix, value_matrix

def calculate_attention_scores(query_matrix: np.ndarray, key_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates attention scores by multiplying the query matrix [seq_len, head_dim] with the transposed key matrix [head_dim, seq_len]
    Resulting attention scores have shape [seq_len, seq_len]
    """
    qkt = np.matmul(query_matrix, key_matrix.T)
    scaling_factor = np.sqrt(query_matrix.shape[1]) #square root of head_dim
    attention_scores = qkt/scaling_factor

    return attention_scores

def mask_attention_scores(attention_scores: np.ndarray) -> np.ndarray:
    """
    Sets future attention scores to negative infinity
    """
    masked_attention_scores = attention_scores.copy()
    ones_matrix = np.ones_like(masked_attention_scores) # creates a matrix the same size as masked_attention_scores with every position set to 1
    binary_mask = np.tril(ones_matrix) # Takes a 2D matrix m and zeroes out everything above the main diagonal, keeping only the lower triangle (hence tri-l)
    masked_attention_scores[binary_mask == 0] = -np.inf

    return masked_attention_scores