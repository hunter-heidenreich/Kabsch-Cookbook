import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from strategies import (
    aligned_pair_3d,
    aligned_pair_nd,
    point_clouds_3d,
    point_clouds_nd,
)

from kabsch_horn import numpy as kabsch_np

_FRAMEWORK_SETTINGS = settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
_NUMPY_SETTINGS = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


class TestRotationInvariants:
    @_FRAMEWORK_SETTINGS
    @given(point_clouds_nd())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_rotation_is_orthogonal_kabsch(
        self, adapter: FrameworkAdapter, P_np: np.ndarray
    ) -> None:
        dim = P_np.shape[-1]
        assume(adapter.supports_dim(dim))
        Q_np = P_np + np.random.default_rng(0).random((1, dim)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.kabsch(P, Q)
        R = adapter.convert_out(res[0])
        np.testing.assert_allclose(R @ R.T, np.eye(dim), atol=adapter.atol * 10)

    @_FRAMEWORK_SETTINGS
    @given(point_clouds_3d())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_rotation_is_orthogonal_horn(
        self, adapter: FrameworkAdapter, P_np: np.ndarray
    ) -> None:
        Q_np = P_np + np.random.default_rng(0).random((1, 3)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.horn(P, Q)
        R = adapter.convert_out(res[0])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=adapter.atol * 10)

    @_FRAMEWORK_SETTINGS
    @given(point_clouds_nd())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_rotation_det_is_positive_kabsch(
        self, adapter: FrameworkAdapter, P_np: np.ndarray
    ) -> None:
        dim = P_np.shape[-1]
        assume(adapter.supports_dim(dim))
        Q_np = P_np + np.random.default_rng(0).random((1, dim)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.kabsch(P, Q)
        R = adapter.convert_out(res[0])
        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=adapter.atol * 10)

    @_FRAMEWORK_SETTINGS
    @given(point_clouds_3d())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_rotation_det_is_positive_horn(
        self, adapter: FrameworkAdapter, P_np: np.ndarray
    ) -> None:
        Q_np = P_np + np.random.default_rng(0).random((1, 3)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.horn(P, Q)
        R = adapter.convert_out(res[0])
        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=adapter.atol * 10)

    @_FRAMEWORK_SETTINGS
    @given(
        st.one_of(
            point_clouds_3d().map(lambda P: (P, "horn")),
            point_clouds_3d().map(lambda P: (P, "horn_with_scale")),
            point_clouds_nd().map(lambda P: (P, "kabsch")),
            point_clouds_nd().map(lambda P: (P, "umeyama")),
        )
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_rmsd_is_nonnegative(
        self, adapter: FrameworkAdapter, input_and_algo: tuple
    ) -> None:
        P_np, algo = input_and_algo
        dim = P_np.shape[-1]
        assume(adapter.supports_dim(dim))
        if algo in ("horn", "horn_with_scale") and dim != 3:
            return
        Q_np = P_np + np.random.default_rng(0).random((1, dim)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        res = func(P, Q)
        rmsd = float(adapter.convert_out(res[-1]))
        assert rmsd >= -adapter.atol * 10

    @_FRAMEWORK_SETTINGS
    @given(point_clouds_nd())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_scale_is_positive_umeyama(
        self, adapter: FrameworkAdapter, P_np: np.ndarray
    ) -> None:
        dim = P_np.shape[-1]
        assume(adapter.supports_dim(dim))
        Q_np = P_np + np.random.default_rng(0).random((1, dim)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.kabsch_umeyama(P, Q)
        c = float(adapter.convert_out(res[2]))
        assert c > -adapter.atol * 10

    @_FRAMEWORK_SETTINGS
    @given(point_clouds_3d())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_scale_is_positive_horn_with_scale(
        self, adapter: FrameworkAdapter, P_np: np.ndarray
    ) -> None:
        Q_np = P_np + np.random.default_rng(0).random((1, 3)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.horn_with_scale(P, Q)
        c = float(adapter.convert_out(res[2]))
        assert c > -adapter.atol * 10


class TestCrossAlgorithmConsistency:
    """Numpy-only cross-algorithm consistency checks (no framework overhead)."""

    @_NUMPY_SETTINGS
    @given(point_clouds_3d())
    def test_kabsch_and_horn_agree_on_rotation_3d(self, P_np: np.ndarray) -> None:
        # Rotation is unique only when point cloud spans all 3 dimensions
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        Q_np = P_np + np.random.default_rng(0).random((1, 3)) * 0.5
        R_k, t_k, rmsd_k = kabsch_np.kabsch(P_np, Q_np)
        R_h, t_h, rmsd_h = kabsch_np.horn(P_np, Q_np)
        np.testing.assert_allclose(R_k, R_h, atol=1e-5)
        np.testing.assert_allclose(t_k, t_h, atol=1e-5)
        np.testing.assert_allclose(float(rmsd_k), float(rmsd_h), atol=1e-5)

    @_NUMPY_SETTINGS
    @given(point_clouds_3d())
    def test_umeyama_and_horn_with_scale_agree_3d(self, P_np: np.ndarray) -> None:
        # Rotation is unique only when point cloud spans all 3 dimensions
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        Q_np = P_np + np.random.default_rng(0).random((1, 3)) * 0.5
        R_u, t_u, c_u, rmsd_u = kabsch_np.kabsch_umeyama(P_np, Q_np)
        R_h, t_h, c_h, rmsd_h = kabsch_np.horn_with_scale(P_np, Q_np)
        np.testing.assert_allclose(R_u, R_h, atol=1e-5)
        np.testing.assert_allclose(t_u, t_h, atol=1e-5)
        np.testing.assert_allclose(float(c_u), float(c_h), atol=1e-5)
        np.testing.assert_allclose(float(rmsd_u), float(rmsd_h), atol=1e-5)

    @_NUMPY_SETTINGS
    @given(aligned_pair_3d())
    def test_kabsch_recovers_known_rotation(self, aligned: tuple) -> None:
        P_np, R_true, t_true, Q_np = aligned
        # Rotation is recoverable only when point cloud spans all 3 dimensions
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        R, t, rmsd = kabsch_np.kabsch(P_np, Q_np)
        np.testing.assert_allclose(R, R_true, atol=1e-6)
        np.testing.assert_allclose(t, t_true, atol=1e-6)
        np.testing.assert_allclose(float(rmsd), 0.0, atol=1e-6)


class TestAlignmentOptimality:
    """Verify that the recovered rotation is locally optimal (numpy-only)."""

    @_NUMPY_SETTINGS
    @given(point_clouds_3d())
    def test_no_rotation_achieves_lower_rmsd(self, P_np: np.ndarray) -> None:
        rng = np.random.default_rng(7)
        Q_np = P_np + rng.random((1, 3)) * 2.0
        R, _t, rmsd_opt = kabsch_np.kabsch(P_np, Q_np)

        # Slightly perturb R with a small random skew and re-orthogonalise
        delta = rng.normal(size=(3, 3)) * 0.01
        R_perturbed, _ = np.linalg.qr(R + delta)
        if np.linalg.det(R_perturbed) < 0:
            R_perturbed[:, 0] *= -1

        P_c = P_np - P_np.mean(0)
        Q_c = Q_np - Q_np.mean(0)
        aligned_perturbed = P_c @ R_perturbed.T
        rmsd_perturbed = float(
            np.sqrt(np.mean(np.sum((aligned_perturbed - Q_c) ** 2, axis=-1)))
        )

        assert float(rmsd_opt) <= rmsd_perturbed + 1e-6


class TestKabschRecoveryND:
    """Verify that kabsch/umeyama recover a known rotation across all frameworks."""

    @_FRAMEWORK_SETTINGS
    @given(aligned_pair_nd())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_kabsch_recovers_known_rotation_nd(
        self, adapter: FrameworkAdapter, aligned: tuple
    ) -> None:
        P_np, R_true, t_true, Q_np, dim = aligned
        assume(adapter.supports_dim(dim))
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, t, rmsd = adapter.kabsch(P, Q)
        R = adapter.convert_out(R)
        t = adapter.convert_out(t)
        np.testing.assert_allclose(R, R_true, atol=adapter.atol * 100)
        np.testing.assert_allclose(t, t_true, atol=adapter.atol * 100)
        np.testing.assert_allclose(
            float(adapter.convert_out(rmsd)), 0.0, atol=adapter.atol * 100
        )

    @_FRAMEWORK_SETTINGS
    @given(aligned_pair_nd())
    @pytest.mark.parametrize("adapter", frameworks)
    def test_kabsch_umeyama_recovers_known_rotation_nd(
        self, adapter: FrameworkAdapter, aligned: tuple
    ) -> None:
        P_np, R_true, t_true, Q_np, dim = aligned
        assume(adapter.supports_dim(dim))
        P_c = P_np - P_np.mean(0)
        # Rotation and scale are recoverable only when the cloud spans all dimensions
        # with sufficient spread; use a strict singular-value threshold
        sv = np.linalg.svd(P_c, compute_uv=False)
        assume(sv[-1] > 1e-3)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, t, c, rmsd = adapter.kabsch_umeyama(P, Q)
        R = adapter.convert_out(R)
        t = adapter.convert_out(t)
        np.testing.assert_allclose(R, R_true, atol=adapter.atol * 100)
        np.testing.assert_allclose(t, t_true, atol=adapter.atol * 100)
        np.testing.assert_allclose(
            float(adapter.convert_out(c)), 1.0, atol=adapter.atol * 100
        )
        np.testing.assert_allclose(
            float(adapter.convert_out(rmsd)), 0.0, atol=adapter.atol * 100
        )
