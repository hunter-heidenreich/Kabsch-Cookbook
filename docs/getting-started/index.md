# Getting Started

## Installation

### pip

```bash
pip install kabsch-horn-cookbook
```

### uv

```bash
uv add kabsch-horn-cookbook
```

### Copy-the-folder

The code has no runtime dependencies beyond the framework itself. Copy the framework folder you need from `src/kabsch_horn/<framework>/` directly into your project:

```
src/kabsch_horn/
├── numpy/
├── pytorch/
├── jax/
├── tensorflow/
└── mlx/
```

Each folder contains two files: `kabsch_svd_nd.py` for SVD-based alignment and `horn_quat_3d.py` for quaternion-based alignment. Copy one file, both, or the whole folder.

## Quickstart

Every framework exports the same API. Replace `pytorch` with `jax`, `tensorflow`, `mlx`, or `numpy` as needed.

### Kabsch alignment (N-dimensional)

```python
import torch
from kabsch_horn import pytorch as kh

# Batched N-dimensional points
P = torch.randn(10, 100, 64)
Q = torch.randn(10, 100, 64)

# R: (10, 64, 64) | t: (10, 64) | rmsd: (10,)
R, t, rmsd = kh.kabsch(P, Q)

# Umeyama variant (with global scale)
# R: (10, 64, 64) | t: (10, 64) | c: (10,) | rmsd: (10,)
R, t, c, rmsd = kh.kabsch_umeyama(P, Q)
```

### Horn alignment (3D quaternion)

```python
P_3d = torch.randn(10, 100, 3)
Q_3d = torch.randn(10, 100, 3)

# R: (10, 3, 3) | t: (10, 3) | rmsd: (10,)
R, t, rmsd = kh.horn(P_3d, Q_3d)

# Horn with scale
R, t, c, rmsd = kh.horn_with_scale(P_3d, Q_3d)
```

### RMSD loss for training

Autodiff frameworks export single-call RMSD loss functions with stable gradients:

```python
loss = kh.kabsch_rmsd(P, Q)
loss.mean().backward()
```

## Framework notes

- **NumPy** -- Forward-pass only. No autograd wrappers or RMSD loss functions.
- **MLX** -- Kabsch is restricted to 3D inputs (`dim == 3`). Horn works for 3D as in all frameworks.
- **JAX** -- Float64 requires `JAX_ENABLE_X64=True` set before importing JAX.

## Compiler support

All functions work with `torch.compile` and `jax.jit`:

=== "PyTorch"

    ```python
    compiled_kabsch = torch.compile(kh.kabsch)
    R, t, rmsd = compiled_kabsch(P, Q)
    ```

=== "JAX"

    ```python
    import jax
    from kabsch_horn import jax as kh

    jitted_kabsch = jax.jit(kh.kabsch)
    R, t, rmsd = jitted_kabsch(P, Q)
    ```
