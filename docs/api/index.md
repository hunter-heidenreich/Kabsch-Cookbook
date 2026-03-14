# API Reference

All frameworks share the same function signatures and return types. Pick the page for your framework below.

## Framework support

| Function | NumPy | PyTorch | JAX | TensorFlow | MLX |
|----------|:-----:|:-------:|:---:|:----------:|:---:|
| `kabsch` | ✓ | ✓ | ✓ | ✓ | ✓ (3D only) |
| `kabsch_umeyama` | ✓ | ✓ | ✓ | ✓ | ✓ (3D only) |
| `horn` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `horn_with_scale` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `kabsch_rmsd` | -- | ✓ | ✓ | ✓ | ✓ |
| `kabsch_umeyama_rmsd` | -- | ✓ | ✓ | ✓ | ✓ |

## Common signatures

**Alignment functions** accept `P` and `Q` tensors of shape `[N, D]` (single) or `[..., N, D]` (batched).

- `kabsch(P, Q)` returns `(R, t, rmsd)` -- rotation `[..., D, D]`, translation `[..., D]`, RMSD `[...]`
- `kabsch_umeyama(P, Q)` returns `(R, t, c, rmsd)` -- adds scale `c: [...]`
- `horn(P, Q)` returns `(R, t, rmsd)` -- 3D only
- `horn_with_scale(P, Q)` returns `(R, t, c, rmsd)` -- 3D only

**RMSD loss functions** (autodiff frameworks only):

- `kabsch_rmsd(P, Q)` returns RMSD scalar(s) with stable gradients
- `kabsch_umeyama_rmsd(P, Q)` returns RMSD scalar(s) with stable gradients

## Framework pages

- [NumPy](numpy.md) -- Forward-pass only, no autograd
- [PyTorch](pytorch.md) -- Full autograd with SafeSVD/SafeEigh
- [JAX](jax.md) -- custom_vjp with safe backward
- [TensorFlow](tensorflow.md) -- GradientTape-compatible
- [MLX](mlx.md) -- Metal-accelerated, 3D only for Kabsch
