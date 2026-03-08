# Changelog

All notable changes to this project are documented here.

## [0.2.0](https://github.com/hunter-heidenreich/Kabsch-Cookbook/compare/v0.1.0...v0.2.0) (2026-03-08)


### Features

* expose __version__ on the kabsch_horn package ([#77](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/77)) ([3586908](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/3586908c3e71b8f966e1ce9b6b0baf0c979e1c65))
* **tooling:** add pre-commit config with hygiene, secret-scan, and ruff hooks ([#86](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/86)) ([fe0e7a3](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/fe0e7a301f09a4d84e28eb8a2717bdcc63e07eb6))


### Bug Fixes

* add [tool.pytest.ini_options] with testpaths to pyproject.toml ([#75](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/75)) ([24ac798](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/24ac798f08e52a80ac31f524f712cd37aa6e333c))
* build __all__ dynamically inside each try block ([#67](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/67)) ([48761d6](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/48761d653a8fc7286f555204342c878b777f7764))
* **ci:** drop push trigger on dev to eliminate duplicate runs ([#87](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/87)) ([f367bf5](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/f367bf527b4af4559e42df51a0b35e5c8be22f1d))
* **ci:** replace uv sync --all-extras --dev with uv sync --dev ([#76](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/76)) ([c454078](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/c454078faaf7c182b3c6586a8b76b33c0df75b96))
* correct GitHub username in pyproject.toml URLs ([#63](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/63)) ([e8cdd9c](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/e8cdd9cf57f02ad2c80f4ca4d85035a256440c2a))
* enforce NaN-propagation contract in test_propagates_nans_gracefully ([#42](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/42)) ([#73](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/73)) ([3d8e62b](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/3d8e62b6734a71afcc80ceb4f56fd4f28867b3ab))
* move shape check before auto-batching unsqueeze in all frameworks ([#69](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/69)) ([f60d0fc](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/f60d0fc080831cb0e9b03265c656199697cac06c)), closes [#49](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/49)
* remove dead _B = P.shape[0] assignments across all frameworks ([#71](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/71)) ([f0cde24](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/f0cde24ca69dd3e0bbb1fa3132272bc8797365ed)), closes [#50](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/50)
* remove dead assignment in PyTorch SVD backward ([#65](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/65)) ([9ccb02a](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/9ccb02a0278b7307cc2b4df9c56bc7866f245e05)), closes [#33](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/33)
* **testing:** extend error handling coverage for horn shape mismatch and dim mismatch ([#84](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/84)) ([3aed5ef](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/3aed5efb3392c25ce5ff42fe86c40cf7ec19a46e))
* **testing:** guard framework imports in adapters.py with try/except ([#68](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/68)) ([68ce6b5](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/68ce6b56373e9f3e3aa52ac61d54740f7d4b66a0)), closes [#46](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/46)
* **testing:** parametrize reference validation tests over multiple seeds ([#83](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/83)) ([8aa00f6](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/8aa00f621f7163fad14ccbd5ae9b67d6c21659c0))
* **testing:** raise Hypothesis example counts and consolidate numpy settings ([#82](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/82)) ([f50ef9e](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/f50ef9e87e02d5637ef414b77b9cf13d4a214556))
* **testing:** remove inline imports and replace class-name dispatch in test_error_handling.py ([#80](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/80)) ([a7775fd](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/a7775fd6a38f430983d8e6c2874ab9e4f780d8d0))
* **tests:** restructure nearly_coplanar_hypothesis to use _inner() pattern ([#98](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/98)) ([8d577cd](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/8d577cd055d0b02e550a7cf9a6b66b2dc8eac518)), closes [#94](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/94) [#95](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/95) [#96](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/96)
* **tests:** strengthen assume guard to require sigma_min &gt; 1e-3 ([#72](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/72)) ([c7c7350](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/c7c735077c9f50faf84236fb1995ed519838771a)), closes [#70](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/70)
* **tests:** use Haar-uniform rotation sampler and add ND recovery tests ([#61](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/61)) ([8c5ea10](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/8c5ea10d376db8c05d63ec5536e6453974e09c7e)), closes [#10](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/10)
* wrap numpy imports in try/except like all other backends ([#64](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/64)) ([a3cecef](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/a3cecef49944745c39d6f1366b5b97525c2fe3bc))


### Documentation

* update numpy docstrings to [..., N, D] shape notation ([#74](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/74)) ([17cc99c](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/17cc99c1edbb682e246e36a8b09eb32eaebf8f41))

## 0.1.0 (2026-03-07)


### Features

* **jax:** Implement JAX backend with correct autodiff and tests ([8c7b4c4](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/8c7b4c4cd2b07c089f679cce3d11c898944b4221))
* **mlx:** Implement MLX backend with custom autodiff and tests ([b9306cf](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/b9306cfda6a482c00d6d30624449732ce1219849))
* setup project correctly and add working PyTorch implementation with SafeSVD ([3cd857d](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/3cd857d6892c9ee429d1ad378d351bd855332644))
* **tf:** Implement TensorFlow backend with correct autodiff and tests ([53e7f97](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/53e7f975809731d665f98b1c6558a5c270aad80c))


### Bug Fixes

* guard MLX import for Linux CI ([#54](https://github.com/hunter-heidenreich/Kabsch-Cookbook/issues/54)) ([2d649c5](https://github.com/hunter-heidenreich/Kabsch-Cookbook/commit/2d649c5ff222d2022c14116cd55923fcdcb1ebaa))
