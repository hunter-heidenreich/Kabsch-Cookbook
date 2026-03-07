# Contributing

Contributions are welcome. This document covers the development workflow.

## Reporting Bugs

Open an issue on GitHub with:
- A minimal reproducible example.
- The framework and version (e.g., PyTorch 2.10, JAX 0.6).
- Expected vs. actual behavior.

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. Install dev dependencies: `uv sync --group dev`
3. Make your changes.
4. Run the test suite: `uv run pytest tests/`
5. Run the linter: `uv run ruff check . && uv run ruff format .`
6. Open a PR against `main` with a clear description of the change.

## Code Style

- Formatting and linting use [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`).
- No em dashes in comments or docstrings. Use `--` if needed.
- Keep docstrings concise and include input/output shapes.
- New framework implementations should follow the existing module structure under `src/kabsch_horn/<framework>/`.

## Adding a New Framework

1. Create `src/kabsch_horn/<framework>/kabsch_svd_nd.py` implementing `kabsch`, `kabsch_umeyama`, `kabsch_rmsd`, and `kabsch_umeyama_rmsd`.
2. Create `src/kabsch_horn/<framework>/horn_quat_3d.py` implementing `horn` and `horn_with_scale`.
3. Create `src/kabsch_horn/<framework>/__init__.py` exporting all public functions.
4. Update `src/kabsch_horn/__init__.py` to re-export with framework suffix.
5. Add an adapter in `tests/adapters.py` and include it in the `frameworks` list.
6. Verify: `uv run pytest tests/` passes for the new framework.

## Running Tests

```bash
# All tests
uv run pytest tests/

# Single file
uv run pytest tests/test_forward_pass_equivalence.py

# Filter by name
uv run pytest tests/ -k "test_identity_mapping"
```

## Releasing

Releases are automated via [release-please](https://github.com/googleapis/release-please).

1. Merge `dev` into `main` using conventional commits (`feat:`, `fix:`, `ci:`, etc.).
2. `release-please` opens or updates a Release PR on `main` automatically, bumping
   `pyproject.toml` and generating `CHANGELOG.md` entries from the commit history.
3. When you are ready to cut a release, merge the Release PR.
4. `release-please` creates the tag and GitHub release automatically.

No manual version bumping or tagging required.
