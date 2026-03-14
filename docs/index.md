# Kabsch-Horn Alignment Cookbook

Differentiable point cloud alignment across five Python frameworks, with gradient-safe backward passes through degenerate inputs.

## Frameworks

| Framework | Kabsch (N-D) | Horn (3D) | Gradient-safe | RMSD loss |
|-----------|:------------:|:---------:|:-------------:|:---------:|
| NumPy     | ✓            | ✓         | --            | --        |
| PyTorch   | ✓            | ✓         | ✓             | ✓         |
| JAX       | ✓            | ✓         | ✓             | ✓         |
| TensorFlow| ✓            | ✓         | ✓             | ✓         |
| MLX       | ✓ (3D only)  | ✓         | ✓             | ✓         |

## Quick example

```python
import torch
from kabsch_horn import pytorch as kh

P = torch.randn(10, 100, 3)
Q = torch.randn(10, 100, 3)

# Kabsch alignment
R, t, rmsd = kh.kabsch(P, Q)

# Single-call RMSD loss for training
loss = kh.kabsch_rmsd(P, Q)
loss.mean().backward()
```

## What's in the cookbook

- **[Getting Started](getting-started/index.md)** -- Installation and quickstart examples.
- **[Algorithms](algorithms/index.md)** -- How Kabsch (SVD) and Horn (quaternion) work, and how gradients are stabilized.
- **[API Reference](api/index.md)** -- Auto-generated docs for every framework.
- **[Guides](guides/index.md)** -- Tutorials and recipes (coming soon).
- **[Benchmarks](benchmarks/index.md)** -- Performance comparisons (coming soon).

## License

MIT -- copy the framework folder you need directly into your project.
