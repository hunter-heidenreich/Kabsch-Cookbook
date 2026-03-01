# Kabsch-Umeyama Alignment Cookbook

A robust, highly tested collection of the **Kabsch** and **Kabsch-Umeyama** optimal structural alignment algorithms, implemented natively across five modern Python math frameworks:

* 🐍 **NumPy**
* 🔥 **PyTorch**
* 🌌 **JAX**
* 🧱 **TensorFlow**
* 🍎 **MLX**

## The "Safe SVD" Gradient Trap

When applying point cloud alignments in machine learning (like calculating Generative flow matching loss functions or evaluating RMSD on 3D molecular structures), normal implementations of Singular Value Decomposition (SVD) will often encounter mathematically degenerate states. 

If multiple singular values are identical (as happens when point clouds possess perfect symmetry), or if certain axes collapse (like mapping coplanar data), standard library gradients derived from the backward SVD pass divide by zero, exploding into `NaN` weights.

**This cookbook fixes this problem directly.** The autograd wrappers provided for PyTorch, JAX, TensorFlow, and MLX all override their SVD computational graphs. We mask identical singular values dynamically during backpropagation, injecting epsilon factors safely bypassing the gradient explosions and leaving your training loops entirely intact.

## Quickstart

This package does *not* force heavy framework dependencies globally. You may copy the explicit framework file you need from `src/kabsch_umeyama` directly into your project, or install this package alongside whatever framework your project uses natively.

### Kabsch versus Umeyama

* **The Kabsch Algorithm** (`kabsch()`): Discovers the optimal 3D valid Rotational alignment matrix ($R$) and subsequent local Translation vector ($t$) between two aligned, identically-sized point clouds ($P$ and $Q$). 
* **The Kabsch-Umeyama Algorithm** (`kabsch_umeyama()`): Enhances Kabsch to calculate uniform dimensional Scales ($c$) alongside $R$ and $t$, mapping smaller structural clusters robustly to enlarged matching variants.

## Usage & API

Each framework exports the same straightforward signature. Batch processing (e.g. `[Batch, Points, 3]`) is universally supported out of the box.

```python
import torch # Or mx, jax, tf, np
from kabsch_umeyama import pytorch as ku

# Shape: (Batch, N, 3) 
P = torch.randn(10, 100, 3) # Target coordinates
Q = torch.randn(10, 100, 3) # Destination coordinates

# Standard Kabsch
# R: (Batch, 3, 3) | t: (Batch, 3) | rmsd: (Batch,)
R, t, rmsd = ku.kabsch(P, Q) 

# Umeyama Algorithm (with global scale)
# R: (Batch, 3, 3) | t: (Batch, 3) | c: (Batch,) | rmsd: (Batch,)
R, t, c, rmsd = ku.kabsch_umeyama(P, Q)

# Fast utility specifically for evaluating RMSD loss
loss = ku.kabsch_rmsd(P, Q)
loss.mean().backward() # Gradients won't explode!
```

## Running Tests

Exhaustive tests span identity checks, known-transform mappings, gradient equivalency calculations (Finite Deflections), and most importantly verifying backpropagation handles mathematical gradient traps (Singular Identical traps, Collinear mapping bounds, coplanar inputs, and pure reflections).

To test every integration simultaneously:

```bash
uv run pytest tests/
```

### Reference Material

Built dynamically adhering to:
* **[Kabsch 1976]** Kabsch, W. (1976). "A solution for the best rotation to relate two sets of vectors."
* **[Kabsch 1978]** Kabsch, W. (1978). "A discussion of the solution for the best rotation to relate two sets of vectors." 
* **[Umeyama 1991]** Umeyama, S. (1991). "Least-squares estimation of transformation parameters between two point patterns." 