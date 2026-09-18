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
    """
    query_matrix = np.matmul(input_embedding, query_weight_matrix)
    key_matrix = np.matmul(input_embedding, key_weight_matrix)
    value_matrix = np.matmul(input_embedding, value_weight_matrix)

    return query_matrix, key_matrix, value_matrix