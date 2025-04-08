# Einops Rearrange

This project provides a minimal implementation of the core functionality of the `einops.rearrange` operation using only NumPy.

## Features

- Pure NumPy implementation
- Supports basic pattern transformations for rearrange (reshape, transpose, flatten)
- Handles ellipsis (`...`) and dynamic dimensions
- Lightweight and easy to understand

## Design Overview

The core function is `rearrange`, which parses the transformation pattern and performs:

1. **Pattern Parsing**: The input and output patterns are parsed into axis groups.
2. **Shape Inference**: Dimension sizes are mapped from input to output, including handling of `...` (ellipsis) and dynamic size (`-1`).
3. **Tensor Operations**: NumPy operations like `reshape`, `transpose`, and `moveaxis` are applied to achieve the desired layout.

Key design decisions include:
- No external dependencies besides NumPy.
- Prioritizing clarity and alignment with how einsum-like patterns work.
- Minimal pattern syntax: supports subspace grouping with parentheses and wildcard ellipsis, but excludes more advanced einsum semantics (e.g. broadcasting or reductions).

## Usage

```python
import numpy as np
from einops import rearrange

x = np.random.rand(2, 3, 4)
y = rearrange(x, 'b c h -> b h c')
print(y.shape)  # (2, 4, 3)
```


