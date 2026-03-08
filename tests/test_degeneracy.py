import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks


class TestDegeneracy:
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("collapse_target", ["P", "Q", "Both"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_origin_collapse_returns_identity_rotation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        collapse_target: str,
    ) -> None:
        """
        Verifies behavior when points collapse to the origin.
        Scale should theoretically go zero (for umeyama).
        """
        dim = 3

        np.random.seed(42)
        P_np = np.random.rand(5, dim).astype(np.float64)
        Q_np = np.random.rand(5, dim).astype(np.float64)

        if collapse_target in ["P", "Both"]:
            P_np = np.zeros_like(P_np)
        if collapse_target in ["Q", "Both"]:
            Q_np = np.zeros_like(Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)
        R_res = adapter.convert_out(res[0])

        # Rotation should be identity to remain well-conditioned
        assert R_res == pytest.approx(np.eye(dim), abs=adapter.atol)

        # Scale check for Umeyama
        if algo == "umeyama":
            c_res = float(adapter.convert_out(res[2]))
            assert np.isfinite(c_res)


@pytest.mark.parametrize("collapse_target", ["P", "Q", "Both"])
@pytest.mark.parametrize("adapter", frameworks)
class TestHornDegeneracy:
    """Forward-pass correctness for Horn under degenerate (collapsed) point clouds."""

    def test_origin_collapse_returns_valid_rotation(
        self,
        adapter: FrameworkAdapter,
        collapse_target: str,
    ) -> None:
        rng = np.random.default_rng(42)
        P_np = rng.standard_normal((20, 3))
        Q_np = rng.standard_normal((20, 3))
        if collapse_target in ["P", "Both"]:
            P_np = np.zeros_like(P_np)
        if collapse_target in ["Q", "Both"]:
            Q_np = np.zeros_like(Q_np)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, _t, rmsd = adapter.horn(P, Q)
        R_np = adapter.convert_out(R)
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert np.isfinite(adapter.convert_out(rmsd))

    def test_origin_collapse_with_scale_returns_finite(
        self,
        adapter: FrameworkAdapter,
        collapse_target: str,
    ) -> None:
        rng = np.random.default_rng(42)
        P_np = rng.standard_normal((20, 3))
        Q_np = rng.standard_normal((20, 3))
        if collapse_target in ["P", "Both"]:
            P_np = np.zeros_like(P_np)
        if collapse_target in ["Q", "Both"]:
            Q_np = np.zeros_like(Q_np)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, _t, c, rmsd = adapter.horn_with_scale(P, Q)
        R_np = adapter.convert_out(R)
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert np.isfinite(adapter.convert_out(c))
        assert np.isfinite(adapter.convert_out(rmsd))


@pytest.mark.parametrize("adapter", frameworks)
class TestHornDegeneracyGeometric:
    """Forward-pass correctness for Horn on collinear and coplanar inputs."""

    def test_collinear_inputs_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
    ) -> None:
        P_np = np.zeros((20, 3))
        P_np[:, 0] = np.linspace(-5, 5, 20)
        Q_np = P_np.copy()
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, _t, rmsd = adapter.horn(P, Q)
        R_np = adapter.convert_out(R)
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert float(adapter.convert_out(rmsd)) == pytest.approx(
            0.0, abs=adapter.atol * 10
        )

    def test_coplanar_inputs_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
    ) -> None:
        rng = np.random.default_rng(7)
        P_np = rng.standard_normal((20, 3))
        P_np[:, 2] = 0.0
        Q_np = P_np.copy()
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, _t, rmsd = adapter.horn(P, Q)
        R_np = adapter.convert_out(R)
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert float(adapter.convert_out(rmsd)) == pytest.approx(
            0.0, abs=adapter.atol * 10
        )

    def test_collinear_inputs_different_clouds_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
    ) -> None:
        P_np = np.zeros((20, 3))
        P_np[:, 0] = np.linspace(-5, 5, 20)
        rng = np.random.default_rng(13)
        Q_np = P_np + rng.standard_normal(P_np.shape) * 0.01
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, _t, rmsd = adapter.horn(P, Q)
        R_np = adapter.convert_out(R)
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert np.isfinite(float(adapter.convert_out(rmsd)))

    def test_coplanar_inputs_different_clouds_return_valid_rotation(
        self,
        adapter: FrameworkAdapter,
    ) -> None:
        rng = np.random.default_rng(7)
        P_np = rng.standard_normal((20, 3))
        P_np[:, 2] = 0.0
        rng2 = np.random.default_rng(21)
        Q_np = P_np + rng2.standard_normal(P_np.shape) * 0.01
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, _t, rmsd = adapter.horn(P, Q)
        R_np = adapter.convert_out(R)
        assert np.allclose(R_np @ R_np.T, np.eye(3), atol=adapter.atol * 10)
        assert np.linalg.det(R_np) > 0
        assert np.isfinite(float(adapter.convert_out(rmsd)))
