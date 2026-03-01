# Kabsch-Horn Alignment Cookbook

A robust, highly tested collection of the **Kabsch** (SVD-based) and **Horn** (Quaternion-based) optimal structural alignment algorithms, implemented natively across five modern Python math frameworks:

* 🐍 **NumPy**
* 🔥 **PyTorch**
* 🌌 **JAX**
* 🧱 **TensorFlow**
* 🍎 **MLX**

## Two Paths to Alignment: SVD vs. Quaternions

This cookbook offers two distinct methodologies for calculating optimal rotation matrices, each with their own strengths:

1. **Kabsch Algorithm (N-Dimensional SVD)**: 
   Traditionally locked to 3D, our SVD implementation has been generalized to support **N-dimensional** latent space alignments. It scales gracefully to higher dimensions (e.g. mapping internal representation alignments).
2. **Horn's Method (3D Quaternions)**: 
   Locked strictly to 3D space, Horn's true closed-form quaternion eigendecomposition avoids the infamous "reflection trap" of SVD entirely without needing ad-hoc determinant sign-flip corrections. It is a mathematically robust alternative for purely 3D point cloud tasks (like molecular conformers or rigid-body physics).

## The "Safe Gradients" Trap

When applying point cloud alignments in machine learning (like calculating Generative flow matching loss functions or evaluating RMSD on 3D molecular structures), normal implementations of Singular Value Decomposition (SVD) and Eigendecomposition will often encounter mathematically degenerate states. 

If eigenvalues or singular values are identical (as happens when point clouds possess perfect symmetry), or if certain axes collapse (like mapping coplanar data), standard library gradients derived from the backward pass divide by zero, exploding into `NaN` weights.

**This cookbook fixes this problem directly.** The autograd wrappers provided for PyTorch, JAX, TensorFlow, and MLX all override their SVD and Eigh computational graphs. We mask identical roots dynamically during backpropagation, injecting epsilon factors safely bypassing the gradient explosions and leaving your training loops entirely intact.

## Quickstart

This package does *not* force heavy framework dependencies globally. You may copy the explicit framework folder you need from `src/kabsch_horn/<framework>/` directly into your project, or install this package alongside whatever framework your project uses natively.

### Core Functions

All algorithms have a standard variant and an **Umeyama** variant. The Umeyama extension enhances the algorithms to calculate a uniform dimensional Scale ($c$) mapping smaller structural clusters robustly to enlarged matching variants.

## Usage & API

Each framework exports the same straightforward signature and automatically routes to the chosen backend. Batch processing (e.g. `[Batch, Points, D]`) is universally supported out of the box.

```python
import torch # Or mx, jax, tf, np
from kabsch_horn import pytorch as ku

# ----------------------------------------------------
# 1. N-Dimensional SVD Kabsch
# ----------------------------------------------------
# N-Dimensional points (e.g., representation matching in 64D)
P_nd = torch.randn(10, 100, 64) 
Q_nd = torch.randn(10, 100, 64)

# R: (Batch, 64, 64) | t: (Batch, 64) | rmsd: (Batch,)
R, t, rmsd = ku.kabsch(P_nd, Q_nd) 

# Umeyama Algorithm (with global scale)
# R: (Batch, 64, 64) | t: (Batch, 64) | c: (Batch,) | rmsd: (Batch,)
R, t, c, rmsd = ku.kabsch_horn(P_nd, Q_nd)

# ----------------------------------------------------
# 2. 3D Closed-Form Quaternion Horn
# ----------------------------------------------------
# 3D points (e.g., standard molecular/physics alignment)
P_3d = torch.randn(10, 100, 3) 
Q_3d = torch.randn(10, 100, 3) 

# R: (Batch, 3, 3) | t: (Batch, 3) | rmsd: (Batch,)
R, t, rmsd = ku.horn(P_3d, Q_3d)

# Umeyama Algorithm for Horn
# R: (Batch, 3, 3) | t: (Batch, 3) | c: (Batch,) | rmsd: (Batch,)
R, t, c, rmsd = ku.horn_umeyama(P_3d, Q_3d)

# Fast utility specifically for evaluating RMSD loss (defaults to SVD Kabsch)
loss = ku.rmsd(P_nd, Q_nd)
loss.mean().backward() # Gradients won't explode!
```

## Running Tests

Exhaustive tests span identity checks, known-transform mappings, gradient equivalency calculations (Finite Deflections), and most importantly verifying backpropagation handles mathematical gradient traps (Singular Identical traps, perfect symmetric cubes, collinear mapping bounds, coplanar inputs, and pure reflections).

To test every integration simultaneously:

```bash
uv run pytest tests/
```

### Reference Material

Built dynamically adhering to:
* **[Kabsch 1976]** Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors."
* **[Kabsch 1978]** Kabsch, W. (1978). "A discussion of the solution for the best rotation to relate two sets of vectors."
* **[Horn 1987]** Horn, B.K.P. (1987). "Closed-form solution of absolute orientation using unit quaternions."
* **[Umeyama 1991]** Umeyama, S. (1991). "Least-squares estimation of transformation parameters between two point patterns."
