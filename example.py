# =====================================================================
# 1. Functional / Stateless Core
# =====================================================================

def scaled_dot_product_attention_np(q, k, v, mask=None):
    """Raw NumPy scaled dot-product calculation."""
    # Simple, pure-function math
    ...

def scaled_dot_product_attention_torch(q, k, v, mask=None):
    """PyTorch scaled dot-product calculation."""
    ...

# =====================================================================
# 2. NumPy Implementation (Explicit Weights + Manual Gradients)
# =====================================================================

class NumPyMultiHeadAttention:
    def __init__(self, d_model, n_heads):
        ...
    def forward(self, x, mask=None):
        ...
    def backward(self, grad_output):
        ...

# =====================================================================
# 3. PyTorch Implementation (nn.Module + Autograd)
# =====================================================================

class TorchMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        ...
    def forward(self, x, mask=None):
        ...

# =====================================================================
# 4. Hugging Face / Wrapper Implementation
# =====================================================================

class HFMultiHeadAttention(nn.Module):
    """Simple wrapper adapting Torch MHA to HuggingFace config/output patterns."""
    ...