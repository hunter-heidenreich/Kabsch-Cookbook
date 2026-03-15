import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from conftest import ALGORITHMS, ALGORITHMS_WITH_SCALE


class TestDegeneracy:
    """Forward-pass correctness under degenerate inputs for all algorithms.

    Horn tests are automatically skipped for dim != 3 via the collection hook.
    """

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("collapse_target", ["P", "Q", "Both"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_origin_collapse_returns_valid_rotation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        collapse_target: str,
        dim: int,
    ) -> None:
        """Verifies behavior when points collapse to the origin."""
        rng = np.random.default_rng(42)
        P_np = rng.random((5, dim)).astype(np.float64)
        Q_np = rng.random((5, dim)).astype(np.float64)

        if collapse_target in ["P", "Both"]:
            P_np = np.zeros_like(P_np)
        if collapse_target in ["Q", "Both"]:
            Q_np = np.zeros_like(Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        R_res = adapter.convert_out(res[0])

        # Rotation should be orthogonal with positive determinant
        # 10x: degenerate cross-covariance yields arbitrary SVD vectors;
        # fallback rotation needs slack
        assert np.allclose(R_res @ R_res.T, np.eye(dim), atol=adapter.atol * 10)
        assert np.linalg.det(R_res) > 0
        assert np.isfinite(adapter.convert_out(res[-1]))  # rmsd

        if algo in ALGORITHMS_WITH_SCALE:
            c_res = float(adapter.convert_out(res[2]))
            assert np.isfinite(c_res)

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("horn", id="horn"),
            pytest.param("horn_with_scale", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_collinear_inputs_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        P_np = np.zeros((20, 3))
        P_np[:, 0] = np.linspace(-5, 5, 20)
        Q_np = P_np.copy()
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        R_np = adapter.convert_out(res[0])
        # 10x: rank-deficient cross-covariance; fallback vectors accumulate
        # rounding in R @ R.T
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        # RMSD for identical clouds: centering cancellation error is O(eps * max_val)
        assert float(adapter.convert_out(res[-1])) == pytest.approx(
            0.0, abs=adapter.atol
        )

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("horn", id="horn"),
            pytest.param("horn_with_scale", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_coplanar_inputs_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        rng = np.random.default_rng(7)
        P_np = rng.standard_normal((20, 3))
        P_np[:, 2] = 0.0
        Q_np = P_np.copy()
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        R_np = adapter.convert_out(res[0])
        # 10x: rank-deficient cross-covariance; fallback vectors accumulate
        # rounding in R @ R.T
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        # RMSD for identical clouds: centering cancellation error is O(eps * max_val)
        assert float(adapter.convert_out(res[-1])) == pytest.approx(
            0.0, abs=adapter.atol
        )

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("horn", id="horn"),
            pytest.param("horn_with_scale", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_collinear_inputs_different_clouds_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        P_np = np.zeros((20, 3))
        P_np[:, 0] = np.linspace(-5, 5, 20)
        rng = np.random.default_rng(13)
        Q_np = P_np + rng.standard_normal(P_np.shape) * 0.01
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        R_np = adapter.convert_out(res[0])
        # 10x: rank-deficient cross-covariance; fallback vectors accumulate
        # rounding in R @ R.T
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert np.isfinite(float(adapter.convert_out(res[-1])))

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("horn", id="horn"),
            pytest.param("horn_with_scale", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_coplanar_inputs_different_clouds_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        rng = np.random.default_rng(7)
        P_np = rng.standard_normal((20, 3))
        P_np[:, 2] = 0.0
        rng2 = np.random.default_rng(21)
        Q_np = P_np + rng2.standard_normal(P_np.shape) * 0.01
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        R_np = adapter.convert_out(res[0])
        # 10x: rank-deficient cross-covariance; fallback vectors accumulate
        # rounding in R @ R.T
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert np.isfinite(float(adapter.convert_out(res[-1])))
