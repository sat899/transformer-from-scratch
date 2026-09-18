# Transformer From Scratch

A small library for building and training transformer models from scratch in **NumPy only**.

It is primarily a learning codebase. The aim is that you can read any file and see exactly what a transformer step is doing.

## Design principles

1. **Functional and modular.** Pieces of the model are small, composable functions (and a few thin helpers) rather than a class hierarchy. Attention, linear maps, activations, loss, and autograd are separate modules you can call independently.

2. **Explicit and easy to read.** Shapes, matrix multiplies, masking, and derivatives are written out in the open. Prefer a few extra lines and comments over hidden framework behaviour.

## File structure

```
src/
├── transformer.py              # Full encoder/decoder stack (assembly)
├── modules/
│   ├── activations.py          # Softmax and other activations
│   ├── attention.py            # Q/K/V, scaled scores, causal mask, attention output
│   ├── embeddings.py           # Token and positional embeddings
│   ├── linear.py               # Linear map y = xW + b
│   └── normalization.py        # LayerNorm / RMSNorm
└── utils/
    ├── autograd.py             # Manual reverse-mode autodiff over recorded ops
    ├── data.py                 # Batching and data helpers
    ├── optimisation.py         # Cross-entropy loss and gradient descent
    └── tokenization.py         # Vocabulary and integer encoding
tests/
├── test_activations.py
└── test_optimisation.py
```

## Tests

Tests use pytest in the `tests/` folder. Test files should start with `test_`, e.g. `test_activations.py`. Tests should print inputs, outputs, and shapes, including intermediate steps.

Run tests:

```
python -m pytest -s
```
