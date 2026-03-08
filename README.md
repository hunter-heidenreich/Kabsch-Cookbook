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

This cookbook addresses this issue directly. The autograd wrappers provided for PyTorch, JAX, TensorFlow, and MLX override their standard SVD and Eigh computational graphs. The implementation dynamically masks identical roots during backpropagation and injects epsilon factors.

Gradient stability is not just claimed -- it is tested. Hypothesis property tests verify that gradients remain finite across coplanar, collinear, reflected, and collapsed inputs. A dedicated descent-direction test verifies that SafeSVD's masked gradients at near-degenerate inputs still reduce RMSD when used in a gradient step, rather than merely being finite. See [`tests/test_differentiability_traps.py`](tests/test_differentiability_traps.py) and [`tests/test_gradient_verification.py`](tests/test_gradient_verification.py).

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

## Verified Properties

The following mathematical properties are validated by property-based tests using [Hypothesis](https://hypothesis.readthedocs.io). Each claim links to the test that justifies it.

### Output invariants

These hold for all frameworks, all precisions, and all valid input shapes.

| Property | Algorithms | Test |
|----------|-----------|------|
| $R R^\top = I$ (orthogonal) | kabsch, horn | [`test_rotation_is_orthogonal_*`](tests/test_properties.py) |
| $\det(R) = +1$ (proper rotation, no reflections) | kabsch, horn | [`test_rotation_det_is_positive_*`](tests/test_properties.py) |
| $\text{RMSD} \geq 0$ | all | [`test_rmsd_is_nonnegative`](tests/test_properties.py) |
| $c > 0$ (scale factor) | kabsch_umeyama, horn_with_scale | [`test_scale_is_positive_*`](tests/test_properties.py) |

### Correctness invariants

These are verified with NumPy over Hypothesis-drawn inputs.

| Property | Test |
|----------|------|
| $\text{RMSD} = \lVert P R^\top + t - Q \rVert_F / \sqrt{N}$ | [`test_rmsd_equals_transform_residual`](tests/test_properties.py) |
| $R$ is globally optimal: no rotation perturbation lowers RMSD | [`test_no_rotation_achieves_lower_rmsd`](tests/test_properties.py) |
| $\text{RMSD}(P, Q) = \text{RMSD}(Q, P)$ | [`test_kabsch_rmsd_is_symmetric`](tests/test_properties.py) |
| $\text{RMSD}(SP + u,\, SQ + u) = \text{RMSD}(P, Q)$ for any rigid transform $(S, u)$ | [`test_rmsd_invariant_to_rigid_transform`](tests/test_properties.py) |
| $R(P + v,\, Q + v) = R(P, Q)$ for any translation $v$ | [`test_r_invariant_to_translation`](tests/test_properties.py) |
| $R(cP,\, cQ) = R(P, Q)$ for any scalar $c > 0$ | [`test_r_invariant_to_uniform_scale`](tests/test_properties.py) |
| When $Q = P R_\text{true}^\top + t$, Umeyama returns $c = 1$ and matches Kabsch $R$, $t$ | [`test_umeyama_equals_kabsch_when_no_scale_change`](tests/test_properties.py) |
| When $Q = c_\text{true}(P R_\text{true}^\top) + t$, Umeyama recovers $c_\text{true}$ exactly | [`test_umeyama_recovers_exact_scale`](tests/test_properties.py) |

### Cross-algorithm consistency

When the cross-covariance $H = (P - \bar{P})^\top (Q - \bar{Q})$ is well-conditioned ($\sigma_{\min}(H) > 10^{-3}$), the SVD and quaternion code paths agree exactly.

| Property | Test |
|----------|------|
| `kabsch` and `horn` return identical $R$, $t$, $\text{RMSD}$ in 3D | [`test_kabsch_and_horn_agree_on_rotation_3d`](tests/test_properties.py) |
| `kabsch_umeyama` and `horn_with_scale` agree in 3D | [`test_umeyama_and_horn_with_scale_agree_3d`](tests/test_properties.py) |

### Gradient stability

SafeSVD and SafeEigh override the standard backward pass to mask near-zero singular value and eigenvalue differences with `eps=1e-12`. The table below lists the degenerate cases explicitly tested.

| Degenerate input | Guarantee | Test |
|-----------------|-----------|------|
| $P = Q$ (identical) | Finite gradient | [`test_gradients_are_stable_when_points_are_identical`](tests/test_differentiability_traps.py) |
| Coplanar points | Finite gradient | [`test_gradients_are_stable_when_points_are_coplanar`](tests/test_differentiability_traps.py) |
| Collinear points | Finite gradient + descent direction | [`test_gradients_are_stable_when_points_are_collinear`](tests/test_differentiability_traps.py) |
| Near-collinear, $P = Q$ (Hypothesis) | Finite gradient | [`test_gradients_stable_nearly_collinear_hypothesis`](tests/test_differentiability_traps.py) |
| Near-collinear, $P \neq Q$ (Hypothesis) | Finite gradient | [`test_gradients_stable_nearly_collinear_different_clouds`](tests/test_differentiability_traps.py) |
| Near-coplanar (Hypothesis, $d \geq 3$) | Finite gradient | [`test_gradients_stable_nearly_coplanar_hypothesis`](tests/test_differentiability_traps.py) |
| Reflection (improper $R$ would be needed) | Finite gradient + $\det(R) = +1$ | [`test_gradients_are_stable_when_points_are_reflected`](tests/test_differentiability_traps.py) |
| Underdetermined ($N < d$) | Finite gradient | [`test_gradients_are_stable_when_system_is_underdetermined`](tests/test_differentiability_traps.py) |
| Collapse to origin | Finite gradient | [`test_gradients_are_stable_when_points_collapse_to_origin`](tests/test_differentiability_traps.py) |
| Near-collinear or coplanar (Hypothesis, descent) | $\text{RMSD}(P - \alpha \nabla, Q) \leq \text{RMSD}(P, Q) + 0.1$ | [`test_safe_svd_gradient_reduces_rmsd_at_hypothesis_near_degenerate`](tests/test_gradient_verification.py) |

"Descent direction" means one gradient step with $\alpha = 0.01$ does not increase RMSD by more than 0.1. The loose tolerance is intentional -- the guarantee is non-increase, not numerical precision. Gradient accuracy against finite differences is verified for float64 in [`test_gradients_match_finite_differences_hypothesis`](tests/test_gradient_verification.py).

### Known algorithm boundaries

Some inputs are fundamentally degenerate. The library does not raise errors in these cases, but users should understand the implications.

**Near-collinear clouds -- rotation is ambiguous.** When $H = (P - \bar{P})^\top (Q - \bar{Q})$ has a near-zero smallest singular value, multiple rotations achieve the same RMSD. SafeSVD returns a valid rotation ($\det(R) = +1$) with a finite gradient, but the direction is arbitrary. Gradient-based optimizers may behave unpredictably in this regime. See [`test_rotation_is_not_unique_when_cross_covariance_is_degenerate`](tests/test_properties.py).

**MLX: 3D inputs only.** MLX uses a hardcoded 3x3 determinant correction and raises `ValueError` for `dim != 3`.

**NumPy: forward pass only.** NumPy provides no autograd wrappers and does not export `kabsch_rmsd` or `kabsch_umeyama_rmsd`.

**float16 / bfloat16: variance division can overflow.** `kabsch_umeyama` and `horn_with_scale` divide by the point cloud variance. This overflows in half precision when inputs are near-collinear or collapsed to the origin. For production half-precision training, cast inputs to float32 before calling alignment functions.

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

The test suite is organized around mathematical claims rather than code coverage. Each test file targets a distinct category of properties.

| File | What it proves |
|------|----------------|
| [`tests/test_forward_pass_equivalence.py`](tests/test_forward_pass_equivalence.py) | Identical outputs across all frameworks and precisions for the same input; correct batching across `[..., N, D]` shapes |
| [`tests/test_properties.py`](tests/test_properties.py) | Output invariants (orthogonality, det=+1, RMSD >= 0), correctness invariants (RMSD definition, optimality, symmetry, rigid-transform invariance), and cross-algorithm consistency (kabsch = horn in 3D) |
| [`tests/test_differentiability_traps.py`](tests/test_differentiability_traps.py) | Gradient finiteness across all documented degenerate cases; descent direction at singularities |
| [`tests/test_gradient_verification.py`](tests/test_gradient_verification.py) | Analytic gradients match finite differences (float64); batched gradients match sequential; SafeSVD descent at near-degenerate inputs; double backward (PyTorch) |
| [`tests/test_degeneracy.py`](tests/test_degeneracy.py) | Forward-pass validity under extreme degeneracy (origin collapse, collinear, coplanar, underdetermined) |
| [`tests/test_catastrophic_cancellation.py`](tests/test_catastrophic_cancellation.py) | Numerical stability at extreme coordinate magnitudes (1e-6 to 1e6) |
| [`tests/test_error_handling.py`](tests/test_error_handling.py) | Correct exceptions for mismatched shapes, wrong dimensions, and invalid inputs |

The suite runs across 4 frameworks x 4 precisions (float16, bfloat16, float32, float64), with MLX restricted to 3D. Hypothesis property tests use configurable example counts; CI runs the defaults.

Run the test suite with:

```bash
uv run pytest tests/
```

## References

* **[Kabsch 1976]** Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors."
* **[Kabsch 1978]** Kabsch, W. (1978). "A discussion of the solution for the best rotation to relate two sets of vectors."
* **[Horn 1987]** Horn, B.K.P. (1987). "Closed-form solution of absolute orientation using unit quaternions."
* **[Umeyama 1991]** Umeyama, S. (1991). "Least-squares estimation of transformation parameters between two point patterns."
