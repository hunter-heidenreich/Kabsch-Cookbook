# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-03-05

Initial public release.

### Added

- Kabsch algorithm (SVD-based, N-dimensional) for NumPy, PyTorch, JAX, TensorFlow, and MLX.
- Kabsch-Umeyama algorithm (with global scale) across all five frameworks.
- Horn's quaternion method (3D only) for NumPy, PyTorch, JAX, TensorFlow, and MLX.
- Horn's quaternion method with scale for all five frameworks.
- Gradient-safe custom autograd wrappers (`SafeSVD`, `SafeEigh`) for PyTorch, JAX, TensorFlow, and MLX, preventing NaN gradients when point clouds are symmetric or degenerate.
- Single-call RMSD loss functions (`kabsch_rmsd`, `kabsch_umeyama_rmsd`) for all autodiff frameworks (PyTorch, JAX, TensorFlow, MLX).
- Batched inputs with arbitrary leading dimensions (`[..., N, D]`).
- Automatic float16/bfloat16 upcasting to float32 with output downcast.
- Comprehensive test suite covering forward-pass equivalence, differentiability traps, gradient verification, catastrophic cancellation, and degeneracy.
- PEP 561 `py.typed` marker for downstream type checker compatibility.
