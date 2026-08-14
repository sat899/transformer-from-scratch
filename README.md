# Transformer From Scratch

This is a repository / library for building and training transformer models from scratch. It is mostly intended to be a wrapper around PyTorch similar to HuggingFace Transformers, but also includes numpy implementations of core functions in some cases.

It has primarily been created for learning purposes, though over time it may evolve into something more usable.

## Architecture Design Principles

Config-Driven Instantiation: Every component accepts a unified configuration object (e.g., TransformerConfig) to ensure consistent dimensions across layers.

Unified Return Types: Components return explicit dataclasses (or named tuples) rather than plain tuples, preventing confusion over tensor ordering (e.g., separating hidden_states from attention_weights).

Composable Sub-Layers: High-level blocks allow swapping out sub-components (e.g., using RMSNorm instead of LayerNorm, or RoPE instead of absolute positional embeddings) via simple configuration flags or class injection.

## File Structure

src/
├── config.py           # Universal configuration dataclass
├── activations.py      # Custom/standard activation layers (GELU, SwiGLU)
├── embeddings.py       # Token & Positional embeddings (Absolute, RoPE, ALiBi)
├── normalization.py    # LayerNorm, RMSNorm implementations
├── attention.py        # Multi-Head, Grouped-Query, and Causal Attention
├── feedforward.py      # Standard MLP and Gated Feed-Forward blocks
├── transformer.py      # Encoder/Decoder blocks and full stack assembly
├── tokenization.py     # Clean wrapper interfaces around tokenizers
├── data.py             # PyTorch Dataset/DataLoader utilities & collators
└── optimisation.py     # Custom loss functions (CrossEntropy with Label Smoothing), AdamW

## Tests

Tests are built using pytest in the tests/ folder. All test files should start with `test_*` e.g. `test_activations.py`. Tests should print inputs, outputs and shapes, including of all intermediate steps.

Run testS:

```
python -m pytest -s
```
