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


_PAIR_SETTINGS = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
_paired_clouds_nd = st.integers(2, 6).flatmap(
    lambda d: st.integers(d + 2, d * 4 + 4).flatmap(
        lambda n: st.tuples(
            point_clouds_nd(dim=d, n_points=n), point_clouds_nd(dim=d, n_points=n)
        )
    )
)


class TestAlignmentInvariants:
    """Mathematical invariants that must hold for all valid inputs (numpy-only)."""

    @_NUMPY_SETTINGS
    @given(_paired_clouds_nd)
    def test_rmsd_equals_transform_residual(self, PQ: tuple) -> None:
        """kabsch RMSD equals the Frobenius residual of the returned transform."""
        P_np, Q_np = PQ
        n = P_np.shape[0]
        R, t, rmsd = kabsch_np.kabsch(P_np, Q_np)
        residual = np.linalg.norm(P_np @ R.T + t - Q_np) / np.sqrt(n)
        np.testing.assert_allclose(float(rmsd), residual, atol=1e-8)

    @_NUMPY_SETTINGS
    @given(_paired_clouds_nd)
    def test_kabsch_rmsd_is_symmetric(self, PQ: tuple) -> None:
        """RMSD is the same when aligning P->Q and Q->P."""
        P_np, Q_np = PQ
        _, _, rmsd_fwd = kabsch_np.kabsch(P_np, Q_np)
        _, _, rmsd_bwd = kabsch_np.kabsch(Q_np, P_np)
        np.testing.assert_allclose(float(rmsd_fwd), float(rmsd_bwd), atol=1e-8)

    @_NUMPY_SETTINGS
    @given(aligned_pair_nd())
    def test_rmsd_invariant_to_rigid_transform(self, aligned: tuple) -> None:
        """RMSD is unchanged when both P and Q undergo the same rigid transform."""
        P_np, _, _, Q_np, dim = aligned
        rng = np.random.default_rng(99)
        A = rng.standard_normal((dim, dim))
        S, _ = np.linalg.qr(A)
        if np.linalg.det(S) < 0:
            S[:, 0] *= -1
        u = rng.standard_normal(dim)
        _, _, rmsd_orig = kabsch_np.kabsch(P_np, Q_np)
        _, _, rmsd_shifted = kabsch_np.kabsch(P_np @ S.T + u, Q_np @ S.T + u)
        np.testing.assert_allclose(float(rmsd_orig), float(rmsd_shifted), atol=1e-6)

    @_PAIR_SETTINGS
    @given(_paired_clouds_nd)
    def test_r_invariant_to_translation(self, PQ: tuple) -> None:
        """Rotation R is unchanged when both P and Q are shifted by the same vector."""
        P_np, Q_np = PQ
        dim = P_np.shape[-1]
        H = (P_np - P_np.mean(0)).T @ (Q_np - Q_np.mean(0))
        sv_H = np.linalg.svd(H, compute_uv=False)
        assume(sv_H[-1] > 0.1)  # rotation is unique only when H is well-conditioned
        v = np.random.default_rng(55).standard_normal(dim)
        R1, _, _ = kabsch_np.kabsch(P_np, Q_np)
        R2, _, _ = kabsch_np.kabsch(P_np + v, Q_np + v)
        np.testing.assert_allclose(R1, R2, atol=1e-6)

    @_PAIR_SETTINGS
    @given(_paired_clouds_nd, st.floats(0.1, 10.0))
    def test_r_invariant_to_uniform_scale(self, PQ: tuple, c: float) -> None:
        """Rotation R is unchanged when both P and Q are scaled by the same scalar."""
        P_np, Q_np = PQ
        H = (P_np - P_np.mean(0)).T @ (Q_np - Q_np.mean(0))
        sv_H = np.linalg.svd(H, compute_uv=False)
        assume(sv_H[-1] > 0.1)  # rotation is unique only when H is well-conditioned
        R1, _, _ = kabsch_np.kabsch(P_np, Q_np)
        R2, _, _ = kabsch_np.kabsch(P_np * c, Q_np * c)
        np.testing.assert_allclose(R1, R2, atol=1e-6)

    @_NUMPY_SETTINGS
    @given(
        aligned_pair_nd(),
        st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False),
    )
    def test_umeyama_recovers_exact_scale(self, aligned: tuple, c_true: float) -> None:
        """kabsch_umeyama recovers the known scale factor exactly."""
        P_np, R_true, t_true, _, _dim = aligned
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        Q_np = c_true * (P_np @ R_true.T) + t_true
        _, _, c, rmsd = kabsch_np.kabsch_umeyama(P_np, Q_np)
        np.testing.assert_allclose(float(c), c_true, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(float(rmsd), 0.0, atol=1e-4)

    @_NUMPY_SETTINGS
    @given(
        aligned_pair_3d(),
        st.floats(0.5, 3.0, allow_nan=False, allow_infinity=False),
    )
    def test_horn_with_scale_recovers_exact_scale(
        self, aligned: tuple, c_true: float
    ) -> None:
        """horn_with_scale recovers the known scale factor exactly."""
        P_np, R_true, t_true, _ = aligned
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        Q_np = c_true * (P_np @ R_true.T) + t_true
        _, _, c, rmsd = kabsch_np.horn_with_scale(P_np, Q_np)
        np.testing.assert_allclose(float(c), c_true, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(float(rmsd), 0.0, atol=1e-4)
