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

**Alignment functions** accept `P` and `Q` tensors of shape `[N, D]` (single) or `[..., N, D]` (batched). All alignment functions accept an optional `weights` parameter of shape `[..., N]` for per-point weighting.

- `kabsch(P, Q, weights=None)` returns `(R, t, rmsd)` -- rotation `[..., D, D]`, translation `[..., D]`, RMSD `[...]`
- `kabsch_umeyama(P, Q, weights=None)` returns `(R, t, c, rmsd)` -- adds scale `c: [...]`
- `horn(P, Q, weights=None)` returns `(R, t, rmsd)` -- 3D only
- `horn_with_scale(P, Q, weights=None)` returns `(R, t, c, rmsd)` -- 3D only

**RMSD loss functions** (autodiff frameworks only):

- `kabsch_rmsd(P, Q, weights=None)` returns RMSD `[...]` with stable gradients
- `kabsch_umeyama_rmsd(P, Q, weights=None)` returns RMSD `[...]` with stable gradients

**Weights parameter:**

- Shape: `[..., N]` -- one weight per point, matching the batch and point dimensions of `P` and `Q`
- Must be non-negative and sum to a positive value along the points axis
- When `None` (default), all points are weighted equally
- When provided, centroids and RMSD use weighted means

## Framework pages

- [NumPy](numpy.md) -- Forward-pass only, no autograd
- [PyTorch](pytorch.md) -- Full autograd with SafeSVD/SafeEigh
- [JAX](jax.md) -- custom_vjp with safe backward
- [TensorFlow](tensorflow.md) -- GradientTape-compatible
- [MLX](mlx.md) -- Metal-accelerated, 3D only for Kabsch
