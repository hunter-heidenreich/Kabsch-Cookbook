# Kabsch-Horn Alignment Cookbook

A collection of the Kabsch (SVD-based) and Horn (Quaternion-based) optimal structural alignment algorithms. These are implemented natively across five Python math frameworks:

* 🐍 **NumPy**
* 🔥 **PyTorch**
* 🌌 **JAX**
* 🧱 **TensorFlow**
* 🍎 **MLX**

## Zero-Dependency Integration

This cookbook is designed to be dropped directly into your codebase without adding external package bloat.

Simply copy the specific framework folder you need from `src/kabsch_horn/<framework>/` directly into your project. Better yet, rip the code, remix it, or rewrite it for your specific use case. The project operates fully under the MIT license, so you are free to borrow, modify, and distribute exactly what you need without strings attached.

## Two Paths to Alignment

This cookbook provides two methodologies for calculating optimal rotation matrices.

### Kabsch Algorithm (N-Dimensional SVD)
Traditionally used for 3D coordinates, this SVD implementation supports N-dimensional latent space alignments. It scales to higher dimensions for tasks like mapping internal representations.

### Horn's Method (3D Quaternions)
Horn's method applies strictly to 3D space. It uses a closed-form quaternion eigendecomposition to calculate alignment. It avoids the reflection trap often encountered in SVD approaches. This makes it a reliable choice for 3D point cloud tasks, such as molecular conformers or rigid-body physics.

## Stabilizing Gradients

Point cloud alignments evaluated during neural network training often encounter mathematically degenerate states. For example, point clouds with perfect symmetry produce identical eigenvalues or singular values. Standard library gradients derived from the backward pass divide by these numerical differences, resulting in zero-division and `NaN` weights.

This cookbook addresses this issue directly. The autograd wrappers provided for PyTorch, JAX, TensorFlow, and MLX override their standard SVD and Eigh computational graphs. The implementation dynamically masks identical roots during backpropagation and injects epsilon factors. This stabilizes the gradients during your training loops.

## Usage

Each framework exports a consistent signature and routes to the chosen backend. Batch processing (`[Batch, Points, D]`) is supported for all functions.

```python
import torch # Or mx, jax, tf, np
from kabsch_horn import pytorch as kh

# 1. N-Dimensional SVD Kabsch
# N-Dimensional points (e.g., representation matching in 64D)
P_nd = torch.randn(10, 100, 64)
Q_nd = torch.randn(10, 100, 64)

# R: (Batch, 64, 64) | t: (Batch, 64) | rmsd: (Batch,)
R, t, rmsd = kh.kabsch(P_nd, Q_nd)

# Umeyama Algorithm (with global scale)
# R: (Batch, 64, 64) | t: (Batch, 64) | c: (Batch,) | rmsd: (Batch,)
R, t, c, rmsd = kh.kabsch_umeyama(P_nd, Q_nd)

# 2. 3D Closed-Form Quaternion Horn
# 3D points (e.g., standard molecular/physics alignment)
P_3d = torch.randn(10, 100, 3)
Q_3d = torch.randn(10, 100, 3)

# R: (Batch, 3, 3) | t: (Batch, 3) | rmsd: (Batch,)
R, t, rmsd = kh.horn(P_3d, Q_3d)

# Umeyama Algorithm for Horn
# R: (Batch, 3, 3) | t: (Batch, 3) | c: (Batch,) | rmsd: (Batch,)
R, t, c, rmsd = kh.horn_with_scale(P_3d, Q_3d)

# Fast utility for evaluating RMSD loss (defaults to SVD Kabsch)
loss = kh.kabsch_rmsd(P_nd, Q_nd)
loss.mean().backward() # Gradients remain stable
```

## Framework Support

The primary functions (`kabsch`, `kabsch_umeyama`, `horn`, and `horn_with_scale`) are supported across PyTorch, JAX, TensorFlow, MLX, and NumPy.

* **PyTorch, JAX, TensorFlow, MLX:**
  These autodiff frameworks use our custom `safe_svd` and `safe_eigh` operations. They return identical forward-pass results to the standard libraries while stabilizing differential calculations during the backward pass. They also provide single-call wrappers (`kh.kabsch_rmsd` and `kh.kabsch_umeyama_rmsd`) for gradient-safe evaluations of global coordinate loss.
* **NumPy:**
  Focuses strictly on pure forward-pass evaluations.

## Extending the Cookbook

### Adding Custom Loss Functions

Each framework's `kabsch_rmsd` and `kabsch_umeyama_rmsd` functions are the simplest entry point for gradient-based training. For more complex losses, call `kabsch` or `horn` directly and operate on the returned `R`, `t`, and `rmsd` tensors:

```python
from kabsch_horn import pytorch as kh

def contrastive_alignment_loss(P_pos, Q_pos, P_neg, Q_neg):
    rmsd_pos = kh.kabsch_rmsd(P_pos, Q_pos)
    rmsd_neg = kh.kabsch_rmsd(P_neg, Q_neg)
    return (rmsd_pos - rmsd_neg + margin).clamp(min=0).mean()
```

The rotation matrix `R` returned by `kabsch` and `horn` is differentiable, so it can be composed into downstream losses (e.g., point-to-point error after applying a learned perturbation on top of `R`).

### Adapting to New Frameworks

To port these algorithms to a new backend, implement the following interface:

1. **`safe_svd(A)`** -- A custom-gradient SVD that masks near-zero singular value differences in the backward pass with `eps=1e-12`. See `src/kabsch_horn/pytorch/kabsch_svd_nd.py` (`SafeSVD`) for the reference implementation.
2. **`safe_eigh(A)`** -- Same pattern for eigendecomposition, used by Horn's method. See `SafeEigh` in `src/kabsch_horn/pytorch/horn_quat_3d.py`.
3. **`kabsch(P, Q)`** -- Accepts `[N, D]` or `[..., N, D]` inputs and returns `(R, t, rmsd)`.
4. **`horn(P, Q)`** -- Accepts `[N, 3]` or `[..., N, 3]` inputs and returns `(R, t, rmsd)`.

The NumPy module (`src/kabsch_horn/numpy/`) is a clean forward-pass-only reference with no autograd dependencies, useful as a starting point.

## Testing

The test suite spans permutations across the supported frameworks. It tests against variable constraints, skipping edge cases where hardware limitations apply.

* **Dimensions**: 2D, 3D, 4D, 10D, and 100D point mapping arrays. MLX limits explicitly to 3D constraints.
* **Precisions**: Verifies structural losses across `float16`, `bfloat16`, `float32`, and `float64`. Tests check precision mantissa truncations.
* **Framework Targets**: Compares outputs across NumPy, PyTorch, JAX, TensorFlow, and MLX within hardware-dictated precision bounds.

The suite verifies:
1. **Forward Pass Equivalence**: Checks identity mapping, exact reconstruction against known transformations, N-dimensional batching evaluation (`[2, 3, N, D]`), and benchmarks standard implementations.
2. **Differentiability**: Validates that autodiff engines maintain numerical stability under mathematically critical states. This includes exactly identical input states, coplanar points, collinear data, perfect symmetrical primitives, mathematical reflections, and origin collapses.
3. **Gradient Verification**: Compares numerical auto-differentiation against Finite Difference evaluations to validate descent directions.
4. **Catastrophic Cancellation**: Tests extreme magnitude shifts to verify float stability.
5. **Degeneracy and Malformed Systems**: Verifies behavior under underdetermined layouts (points < dims) or mismatched arrays.

Run the test suite with:

```bash
uv run pytest tests/
```

## References

* **[Kabsch 1976]** Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors."
* **[Kabsch 1978]** Kabsch, W. (1978). "A discussion of the solution for the best rotation to relate two sets of vectors."
* **[Horn 1987]** Horn, B.K.P. (1987). "Closed-form solution of absolute orientation using unit quaternions."
* **[Umeyama 1991]** Umeyama, S. (1991). "Least-squares estimation of transformation parameters between two point patterns."
