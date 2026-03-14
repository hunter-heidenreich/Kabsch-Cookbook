# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch Strategy

- `main` -- production branch; only receives merges from `dev` when ready for release
- `dev` -- staging branch for prod-ready work; all PRs from feature branches target `dev`
- New features and fixes: create a branch off `dev`, then open a PR against `dev`

## Commands

```bash
# Run tests (fast, default -- skips float16/bfloat16 non-dtype tests, fewer Hypothesis examples)
uv run pytest tests/

# Run full exhaustive test suite (CI)
uv run pytest tests/ --full

# Run a single test file
uv run pytest tests/test_forward_pass_equivalence.py

# Run a specific test by name
uv run pytest tests/ -k "test_identity_mapping"

# Lint and format
uv run ruff check .
uv run ruff format .
```

The project uses `uv` for dependency management and task running. All dev dependencies (torch, jax, tensorflow, mlx, numpy, pytest, ruff) are in `[dependency-groups] dev` in `pyproject.toml`.

## Architecture

**Source layout**: `src/kabsch_horn/<framework>/` with one subpackage per framework: `numpy`, `pytorch`, `jax`, `tensorflow`, `mlx`. Each framework subpackage has two modules:
- `kabsch_svd_nd.py` - Kabsch and Umeyama alignment via SVD (N-dimensional)
- `horn_quat_3d.py` - Horn's quaternion alignment (3D only)

**Stable gradient wrappers**: The key innovation is custom autograd functions (`SafeSVD`, `SafeEigh`) defined in each autodiff framework's module. These override the backward pass to mask near-zero eigenvalue/singular-value differences with `eps`, preventing NaN gradients when point clouds are symmetric or degenerate. NumPy has no such wrapper (forward-pass only).

**Public API per framework**: Each `__init__.py` exports `kabsch`, `kabsch_umeyama`, `horn`, `horn_with_scale`. Autodiff frameworks also export `kabsch_rmsd` and `kabsch_umeyama_rmsd` (single-call gradient-safe RMSD loss functions). The top-level `kabsch_horn/__init__.py` re-exports all symbols with framework suffixes (e.g., `kabsch_torch`, `horn_jax`), with silent `ImportError` fallback if a framework is not installed.

**Tensor conventions**:
- Input shape: `[N, D]` (single) or `[..., N, D]` (batched, arbitrary leading dims)
- `kabsch`/`horn` return `(R, t, rmsd)` where `R: [..., D, D]`, `t: [..., D]`, `rmsd: [...]`
- `kabsch_umeyama`/`horn_with_scale` return `(R, t, c, rmsd)` with scale `c: [...]`
- MLX adapter is restricted to `dim == 3` (hardcoded 3x3 determinant correction)
- `float16`/`bfloat16` inputs are internally upcast to `float32` then downcast on output

## Testing

**Test structure**:
- `tests/adapters.py` - `FrameworkAdapter` base class + per-framework subclasses (`PyTorchAdapter`, `JAXAdapter`, `TFAdapter`, `MLXAdapter`). Adapters unify `convert_in`, `convert_out`, `get_grad`, and tolerance levels across frameworks and precisions.
- `tests/conftest.py` - Shared pytest fixtures (`dim`, `identity_points`, `known_transform_points`, `coplanar_points`, etc.) and a `pytest_collection_modifyitems` hook that filters out tests where the adapter's `supports_dim()` returns False (e.g., MLX only runs 3D tests).
- `tests/utils.py` - `compute_numeric_grad` (finite-difference gradient checker) and `check_transform_close` helper.

**Test files**: `test_forward_pass_equivalence.py`, `test_differentiability_traps.py`, `test_gradient_verification.py`, `test_catastrophic_cancellation.py`, `test_degeneracy.py`, `test_error_handling.py`.

**JAX note**: `conftest.py` sets `JAX_ENABLE_X64=True` to allow float64. This must remain as the first env-var set before jax imports.

## Writing Style (for docs/comments)

Per `.github/copilot-instructions.md`:
- No em dashes (`--` is fine, `--` not `--`)
- No negation/contrastive reframes ("not X, but Y")
- Clear and concise for a broad ML audience
