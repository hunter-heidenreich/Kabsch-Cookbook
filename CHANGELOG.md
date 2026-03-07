# Changelog

All notable changes to this project are documented here.

## 0.1.0 (2026-03-07)


### Features

* **jax:** Implement JAX backend with correct autodiff and tests ([8c7b4c4](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/8c7b4c4cd2b07c089f679cce3d11c898944b4221))
* **mlx:** Implement MLX backend with custom autodiff and tests ([b9306cf](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/b9306cfda6a482c00d6d30624449732ce1219849))
* setup project correctly and add working PyTorch implementation with SafeSVD ([3cd857d](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/3cd857d6892c9ee429d1ad378d351bd855332644))
* **tf:** Implement TensorFlow backend with correct autodiff and tests ([53e7f97](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/53e7f975809731d665f98b1c6558a5c270aad80c))


### Bug Fixes

* guard MLX import for Linux CI ([#54](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/54)) ([2d649c5](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/2d649c5ff222d2022c14116cd55923fcdcb1ebaa))

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
