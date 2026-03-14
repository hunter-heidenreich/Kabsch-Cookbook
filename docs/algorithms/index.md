# Algorithms

## Two paths to alignment

Given two point clouds \(P\) and \(Q\) with \(N\) points in \(D\) dimensions, the goal is to find a rotation \(R\), translation \(t\), and optionally a scale \(c\) that minimizes:

\[
\text{RMSD} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| c \cdot R p_i + t - q_i \|^2}
\]

This library provides two algorithms for solving this problem.

### Kabsch algorithm (N-dimensional SVD)

The Kabsch algorithm works in any number of dimensions. It centers both clouds, computes the cross-covariance matrix \(H = (P - \bar{P})^\top (Q - \bar{Q})\), and takes the SVD:

\[
H = U \Sigma V^\top
\]

The optimal rotation is \(R = V \, \text{diag}(1, \ldots, 1, \det(V U^\top)) \, U^\top\), where the determinant correction ensures \(\det(R) = +1\) (proper rotation, no reflections).

The Umeyama extension adds a global scale factor \(c\) computed from the ratio of aligned variance to source variance.

### Horn's method (3D quaternions)

Horn's method applies strictly to 3D. It constructs a 4x4 symmetric matrix from the cross-covariance and finds the eigenvector corresponding to the largest eigenvalue. This eigenvector is a unit quaternion representing the optimal rotation. The closed-form quaternion approach avoids the reflection trap that SVD methods must handle with a determinant sign correction.

## Stabilizing gradients

Point cloud alignments during neural network training often encounter degenerate states -- symmetric or near-collinear inputs that produce identical eigenvalues or singular values. Standard library gradients divide by differences of these values, causing zero-division and NaN weights.

### SafeSVD and SafeEigh

The autograd wrappers for PyTorch, JAX, TensorFlow, and MLX override the standard backward pass. During backpropagation, they mask near-zero differences between singular values (or eigenvalues) with \(\varepsilon = \text{finfo}(\text{dtype}).\text{eps}\):

\[
\frac{1}{\sigma_i - \sigma_j} \rightarrow \begin{cases} \frac{1}{\sigma_i - \sigma_j} & \text{if } |\sigma_i - \sigma_j| > \varepsilon \\ 0 & \text{otherwise} \end{cases}
\]

This preserves gradient accuracy for well-conditioned inputs while preventing NaN propagation at degeneracies.

### What the masking guarantees

| Degenerate input | Guarantee |
|-----------------|-----------|
| Identical point clouds (\(P = Q\)) | Finite gradient |
| Coplanar points | Finite gradient |
| Collinear points | Finite gradient + descent direction |
| Near-collinear, near-coplanar (Hypothesis-tested) | Finite gradient |
| Reflection (improper \(R\) would be needed) | Finite gradient, \(\det(R) = +1\) |
| Underdetermined (\(N < D\)) | Finite gradient |
| Collapse to origin | Finite gradient |

"Descent direction" means one gradient step with \(\alpha = 0.01\) does not increase RMSD by more than 0.1.

## Verified properties

### Output invariants

These hold for all frameworks, all precisions, and all valid input shapes.

| Property | Algorithms |
|----------|-----------|
| \(R R^\top = I\) (orthogonal) | kabsch, horn |
| \(\det(R) = +1\) (proper rotation) | kabsch, horn |
| \(\text{RMSD} \geq 0\) | all |
| \(c > 0\) (scale factor) | kabsch_umeyama, horn_with_scale |

### Correctness invariants

| Property |
|----------|
| \(\text{RMSD} = \lVert P R^\top + t - Q \rVert_F / \sqrt{N}\) |
| \(R\) is globally optimal: no rotation perturbation lowers RMSD |
| \(\text{RMSD}(P, Q) = \text{RMSD}(Q, P)\) |
| \(\text{RMSD}(SP + u, SQ + u) = \text{RMSD}(P, Q)\) for rigid \((S, u)\) |
| \(R(P + v, Q + v) = R(P, Q)\) for any translation \(v\) |
| \(R(cP, cQ) = R(P, Q)\) for scalar \(c > 0\) |

### Cross-algorithm consistency

When the cross-covariance \(H\) is well-conditioned (\(\sigma_{\min}(H) > 10^{-3}\)), the SVD and quaternion code paths agree exactly in 3D.

## Known boundaries

!!! warning "Near-collinear clouds"
    When \(H\) has a near-zero smallest singular value, multiple rotations achieve the same RMSD. SafeSVD returns a valid rotation with a finite gradient, but the direction is arbitrary.

!!! warning "float16/bfloat16 overflow"
    `kabsch_umeyama` and `horn_with_scale` divide by point cloud variance. This can overflow in half precision with near-collinear or collapsed inputs. Prefer float32 or higher for training.

!!! info "JAX double backward"
    `jax.grad(jax.grad(f))` through the Kabsch code path raises `NotImplementedError`. Horn (eigh-based) supports double backward in JAX without issue.
