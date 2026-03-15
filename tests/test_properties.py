import os

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from conftest import ALGORITHMS, ALGORITHMS_3D_ONLY, ALGORITHMS_WITH_SCALE
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from strategies import (
    aligned_pair_3d,
    aligned_pair_nd,
    nearly_collinear_3d,
    point_clouds_3d,
    point_clouds_nd,
)

from kabsch_horn import numpy as kabsch_np

_FAST = os.environ.get("KABSCH_TEST_FAST") == "1"

_FRAMEWORK_SETTINGS = settings(
    max_examples=20 if _FAST else 100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
_NUMPY_SETTINGS = settings(
    max_examples=50 if _FAST else 200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


class TestRotationInvariants:
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    @_FRAMEWORK_SETTINGS
    @given(data=st.data())
    def test_rotation_is_orthogonal(
        self, algo: str, adapter: FrameworkAdapter, data
    ) -> None:
        dims = adapter.supported_dims()
        if algo in ALGORITHMS_3D_ONLY:
            dims = [d for d in dims if d == 3]
        aligned = data.draw(aligned_pair_nd(dims=dims))
        P_np, _R_true, _t_true, Q_np, dim = aligned
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        res = func(P, Q)
        R = adapter.convert_out(res[0])
        # 10x: each entry of R @ R.T is a D-term dot product with O(D * eps) error,
        # layered on top of SVD rounding already in atol
        np.testing.assert_allclose(R @ R.T, np.eye(dim), atol=adapter.atol * 10)

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    @_FRAMEWORK_SETTINGS
    @given(data=st.data())
    def test_rotation_det_is_positive(
        self, algo: str, adapter: FrameworkAdapter, data
    ) -> None:
        dims = adapter.supported_dims()
        if algo in ALGORITHMS_3D_ONLY:
            dims = [d for d in dims if d == 3]
        aligned = data.draw(aligned_pair_nd(dims=dims))
        P_np, _R_true, _t_true, Q_np, _dim = aligned
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        res = func(P, Q)
        R = adapter.convert_out(res[0])
        # 10x: determinant accumulates O(D) multiplications of rounded entries
        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=adapter.atol * 10)

    @pytest.mark.parametrize("adapter", frameworks)
    @_FRAMEWORK_SETTINGS
    @given(data=st.data())
    def test_rmsd_is_nonnegative(self, adapter: FrameworkAdapter, data) -> None:
        dims = adapter.supported_dims()
        algo = data.draw(st.sampled_from(ALGORITHMS))
        if algo in ALGORITHMS_3D_ONLY:
            use_dims = [d for d in dims if d == 3]
        else:
            use_dims = dims
        P_np = data.draw(point_clouds_nd(dims=use_dims))
        dim = P_np.shape[-1]
        Q_np = P_np + np.random.default_rng(0).random((1, dim)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        res = func(P, Q)
        rmsd = float(adapter.convert_out(res[-1]))
        assert rmsd >= 0

    @pytest.mark.parametrize("algo", list(ALGORITHMS_WITH_SCALE))
    @pytest.mark.parametrize("adapter", frameworks)
    @_FRAMEWORK_SETTINGS
    @given(data=st.data())
    def test_scale_is_positive(
        self, algo: str, adapter: FrameworkAdapter, data
    ) -> None:
        dims = adapter.supported_dims()
        if algo in ALGORITHMS_3D_ONLY:
            dims = [d for d in dims if d == 3]
        P_np = data.draw(point_clouds_nd(dims=dims))
        dim = P_np.shape[-1]
        Q_np = P_np + np.random.default_rng(0).random((1, dim)) * 0.5
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        res = func(P, Q)
        c = float(adapter.convert_out(res[2]))
        assert c >= 0


class TestCrossAlgorithmConsistency:
    """Numpy-only cross-algorithm consistency checks (no framework overhead)."""

    @_NUMPY_SETTINGS
    @given(
        point_clouds_3d(),
        arrays(
            np.float64,
            (3,),
            elements=st.floats(0, 5, allow_nan=False, allow_infinity=False),
        ),
    )
    def test_kabsch_and_horn_agree_on_rotation_3d(
        self, P_np: np.ndarray, shift: np.ndarray
    ) -> None:
        # Rotation is unique only when point cloud spans all 3 dimensions
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        Q_np = P_np + shift
        R_k, t_k, rmsd_k = kabsch_np.kabsch(P_np, Q_np)
        R_h, t_h, rmsd_h = kabsch_np.horn(P_np, Q_np)
        # Cross-algorithm: SVD-based kabsch vs eigen-based horn may diverge by
        # O(eps * cond(H)^2); with sv[-1] > 1e-3, cond < 1000 so error < 1e-9
        np.testing.assert_allclose(R_k, R_h, atol=1e-8)
        np.testing.assert_allclose(t_k, t_h, atol=1e-8)
        np.testing.assert_allclose(float(rmsd_k), float(rmsd_h), atol=1e-8)

    @_NUMPY_SETTINGS
    @given(
        point_clouds_3d(),
        arrays(
            np.float64,
            (3,),
            elements=st.floats(0, 5, allow_nan=False, allow_infinity=False),
        ),
    )
    def test_umeyama_and_horn_with_scale_agree_3d(
        self, P_np: np.ndarray, shift: np.ndarray
    ) -> None:
        # Rotation is unique only when point cloud spans all 3 dimensions
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        Q_np = P_np + shift
        R_u, t_u, c_u, rmsd_u = kabsch_np.kabsch_umeyama(P_np, Q_np)
        R_h, t_h, c_h, rmsd_h = kabsch_np.horn_with_scale(P_np, Q_np)
        # Cross-algorithm: SVD-based umeyama vs eigen-based horn_with_scale;
        # with sv[-1] > 1e-3, cond < 1000 so error < 1e-9
        np.testing.assert_allclose(R_u, R_h, atol=1e-8)
        np.testing.assert_allclose(t_u, t_h, atol=1e-8)
        np.testing.assert_allclose(float(c_u), float(c_h), atol=1e-8)
        np.testing.assert_allclose(float(rmsd_u), float(rmsd_h), atol=1e-8)

    @_NUMPY_SETTINGS
    @given(aligned_pair_nd())
    def test_umeyama_equals_kabsch_when_no_scale_change(self, aligned: tuple) -> None:
        """When true scale is 1, Umeyama's extra DOF should not hurt: R, t match Kabsch.

        aligned_pair_nd() produces Q = P @ R_true.T + t with no scale change, so the
        optimal Umeyama scale is 1. This test makes that cross-algorithm consistency
        property explicit and citable.
        """
        P_np, _, _, Q_np, _ = aligned
        P_c = P_np - P_np.mean(0)
        sv = np.linalg.svd(P_c, compute_uv=False)
        assume(sv[-1] > 1e-3)
        R_k, t_k, _ = kabsch_np.kabsch(P_np, Q_np)
        R_u, t_u, c_u, _ = kabsch_np.kabsch_umeyama(P_np, Q_np)
        # Same SVD path; only the trivial scale=1 computation adds rounding
        np.testing.assert_allclose(R_u, R_k, atol=1e-10)
        np.testing.assert_allclose(t_u, t_k, atol=1e-10)
        np.testing.assert_allclose(float(c_u), 1.0, atol=1e-10)

    @_NUMPY_SETTINGS
    @given(aligned_pair_3d())
    def test_kabsch_recovers_known_rotation(self, aligned: tuple) -> None:
        P_np, R_true, t_true, Q_np = aligned
        # Rotation is recoverable only when point cloud spans all 3 dimensions
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)
        R, t, rmsd = kabsch_np.kabsch(P_np, Q_np)
        # SVD on well-conditioned H (sv[-1] > 1e-3) recovers R to ~eps * cond(H) ~ 1e-11
        np.testing.assert_allclose(R, R_true, atol=1e-10)
        np.testing.assert_allclose(t, t_true, atol=1e-10)
        np.testing.assert_allclose(float(rmsd), 0.0, atol=1e-10)


class TestAlignmentOptimality:
    """Verify that the recovered rotation is locally optimal (numpy-only)."""

    @_NUMPY_SETTINGS
    @given(aligned_pair_3d())
    def test_returned_rotation_is_locally_optimal_kabsch(self, aligned: tuple) -> None:
        P_np, _R_true, _t_true, Q_np = aligned
        rng = np.random.default_rng(7)
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

        # Optimal R must beat any perturbed R; 1e-6 absorbs float64 rounding
        assert float(rmsd_opt) <= rmsd_perturbed + 1e-6


class TestKabschRecoveryND:
    """Verify that kabsch/umeyama recover a known rotation across all frameworks."""

    @pytest.mark.parametrize("adapter", frameworks)
    @_FRAMEWORK_SETTINGS
    @given(data=st.data())
    def test_kabsch_recovers_known_rotation_nd(
        self, adapter: FrameworkAdapter, data
    ) -> None:
        aligned = data.draw(aligned_pair_nd(dims=adapter.supported_dims()))
        P_np, R_true, t_true, Q_np, _dim = aligned
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 0.1)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, t, rmsd = adapter.kabsch(P, Q)
        R = adapter.convert_out(R)
        t = adapter.convert_out(t)
        # 10x: SVD + reconstruct compounds rounding; with sv[-1] > 0.1 (cond < 500),
        # pipeline error is O(D * eps * cond) ~ 10x atol
        assert R == pytest.approx(R_true, rel=adapter.rtol, abs=adapter.atol * 10)
        assert t == pytest.approx(t_true, rel=adapter.rtol, abs=adapter.atol * 10)
        assert float(adapter.convert_out(rmsd)) == pytest.approx(
            0.0, rel=adapter.rtol, abs=adapter.atol * 10
        )

    @pytest.mark.parametrize("adapter", frameworks)
    @_FRAMEWORK_SETTINGS
    @given(data=st.data())
    def test_kabsch_umeyama_recovers_known_rotation_nd(
        self, adapter: FrameworkAdapter, data
    ) -> None:
        aligned = data.draw(aligned_pair_nd(dims=adapter.supported_dims()))
        P_np, R_true, t_true, Q_np, _dim = aligned
        P_c = P_np - P_np.mean(0)
        # Rotation and scale are recoverable only when the cloud spans all dimensions
        # with sufficient spread; use a strict singular-value threshold
        sv = np.linalg.svd(P_c, compute_uv=False)
        assume(sv[-1] > 0.1)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        R, t, c, rmsd = adapter.kabsch_umeyama(P, Q)
        R = adapter.convert_out(R)
        t = adapter.convert_out(t)
        # 10x: SVD + reconstruct compounds rounding; with sv[-1] > 0.1 (cond < 500),
        # pipeline error is O(D * eps * cond) ~ 10x atol; scale DOF adds minor rounding
        assert R == pytest.approx(R_true, rel=adapter.rtol, abs=adapter.atol * 10)
        assert t == pytest.approx(t_true, rel=adapter.rtol, abs=adapter.atol * 10)
        assert float(adapter.convert_out(c)) == pytest.approx(
            1.0, rel=adapter.rtol, abs=adapter.atol * 10
        )
        assert float(adapter.convert_out(rmsd)) == pytest.approx(
            0.0, rel=adapter.rtol, abs=adapter.atol * 10
        )


@st.composite
def _paired_clouds_nd_composite(draw):
    """Two independent N-D point clouds of matching shape, drawn uniformly."""
    d = draw(st.integers(2, 6))
    n = draw(st.integers(d + 2, d * 4 + 4))
    P = draw(point_clouds_nd(dim=d, n_points=n))
    Q = draw(point_clouds_nd(dim=d, n_points=n))
    return P, Q, d


@st.composite
def _correlated_paired_clouds_nd(draw):
    """N-D point cloud pair with well-conditioned cross-covariance.

    Q = P + noise ensures H = P_c.T @ Q_c ≈ P_c.T @ P_c, which is
    well-conditioned whenever P spans all dimensions.  The assume guard
    rejects the rare case where P itself is degenerate (e.g. all zeros
    after Hypothesis shrinking); with random [-10,10] points this fires
    < 1% of the time, well below Hypothesis's filter_too_much threshold.
    """
    d = draw(st.integers(2, 6))
    n = draw(st.integers(d + 2, d * 4 + 4))
    P = draw(point_clouds_nd(dim=d, n_points=n))
    noise = draw(
        arrays(
            np.float64,
            (n, d),
            elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
        )
    )
    Q = P + noise
    # Reject degenerate P (all-zero or collinear after centering)
    P_c = P - P.mean(0)
    sv = np.linalg.svd(P_c, compute_uv=False)
    assume(sv[-1] > 1e-3)
    return P, Q, d


@st.composite
def _correlated_with_shift(draw):
    """Correlated N-D point cloud pair plus a drawn translation shift vector."""
    P, Q, d = draw(_correlated_paired_clouds_nd())
    v = draw(
        arrays(
            np.float64,
            (d,),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        )
    )
    return P, Q, d, v


@st.composite
def _paired_with_shift(draw):
    """Two independent N-D point clouds plus a drawn translation shift vector."""
    P, Q, d = draw(_paired_clouds_nd_composite())
    v = draw(
        arrays(
            np.float64,
            (d,),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        )
    )
    return P, Q, d, v


class TestAlignmentInvariants:
    """Mathematical invariants that must hold for all valid inputs (numpy-only)."""

    @_NUMPY_SETTINGS
    @given(_paired_clouds_nd_composite())
    def test_rmsd_equals_transform_residual(self, PQ: tuple) -> None:
        """kabsch RMSD equals the Frobenius residual of the returned transform."""
        P_np, Q_np, _ = PQ
        n = P_np.shape[0]
        R, t, rmsd = kabsch_np.kabsch(P_np, Q_np)
        residual = np.linalg.norm(P_np @ R.T + t - Q_np) / np.sqrt(n)

        # Float64 RMSD vs residual: paths differ by ~eps * sqrt(N) * scale
        np.testing.assert_allclose(float(rmsd), residual, atol=1e-12)

    @_NUMPY_SETTINGS
    @given(_paired_clouds_nd_composite())
    def test_kabsch_rmsd_is_symmetric(self, PQ: tuple) -> None:
        """RMSD is the same when aligning P->Q and Q->P."""
        P_np, Q_np, _ = PQ
        _, _, rmsd_fwd = kabsch_np.kabsch(P_np, Q_np)
        _, _, rmsd_bwd = kabsch_np.kabsch(Q_np, P_np)

        # Float64 RMSD vs residual: paths differ by ~eps * sqrt(N) * scale
        np.testing.assert_allclose(float(rmsd_fwd), float(rmsd_bwd), atol=1e-12)

    @_NUMPY_SETTINGS
    @given(_paired_clouds_nd_composite(), st.integers(0, 2**31 - 1))
    def test_rmsd_invariant_to_rigid_transform(self, PQ: tuple, rng_seed: int) -> None:
        """RMSD is unchanged when both P and Q undergo the same rigid transform."""
        P_np, Q_np, dim = PQ
        # No conditioning filter: RMSD depends only on the singular values of
        # H = P_c.T @ Q_c, which are preserved under orthogonal conjugation
        # (S @ H @ S.T has the same SVs as H).  The invariant holds for all
        # inputs, including rank-deficient and zero cross-covariance.
        rng = np.random.default_rng(rng_seed)
        A = rng.standard_normal((dim, dim))
        S, _ = np.linalg.qr(A)
        if np.linalg.det(S) < 0:
            S[:, 0] *= -1
        u = rng.standard_normal(dim)
        _, _, rmsd_orig = kabsch_np.kabsch(P_np, Q_np)
        _, _, rmsd_shifted = kabsch_np.kabsch(P_np @ S.T + u, Q_np @ S.T + u)
        # RMSD depends on SVs of H, which are preserved under orthogonal conjugation;
        # centering after rotation introduces O(eps * ||u|| * N) rounding
        np.testing.assert_allclose(float(rmsd_orig), float(rmsd_shifted), atol=1e-9)

    @_NUMPY_SETTINGS
    @given(_correlated_with_shift())
    def test_r_invariant_to_translation(self, PQdv: tuple) -> None:
        """Rotation R is unchanged when both P and Q are shifted by the same vector."""
        P_np, Q_np, _dim, v = PQdv
        # Rotation is unique only when H is well-conditioned; correlated
        # strategy (Q = P + noise) guarantees this by construction.
        R1, _, _ = kabsch_np.kabsch(P_np, Q_np)
        R2, _, _ = kabsch_np.kabsch(P_np + v, Q_np + v)
        # Same centered data -> same H -> same SVD -> same R;
        # only centering rounding differs
        np.testing.assert_allclose(R1, R2, atol=1e-10)

    @_NUMPY_SETTINGS
    @given(_correlated_paired_clouds_nd(), st.floats(0.1, 10.0))
    def test_r_invariant_to_uniform_scale(self, PQ: tuple, c: float) -> None:
        """Rotation R is unchanged when both P and Q are scaled by the same scalar."""
        P_np, Q_np, _ = PQ
        # Rotation is unique only when H is well-conditioned; correlated
        # strategy (Q = P + noise) guarantees this by construction.
        R1, _, _ = kabsch_np.kabsch(P_np, Q_np)
        R2, _, _ = kabsch_np.kabsch(P_np * c, Q_np * c)
        # Scaling both clouds by c scales H by c^2 but preserves SVD directions
        np.testing.assert_allclose(R1, R2, atol=1e-10)

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
        # Scale = trace(D @ S) / var(P); well-conditioned SVD recovers to ~eps * cond
        np.testing.assert_allclose(float(c), c_true, rtol=1e-8, atol=1e-8)
        # RMSD ~ 0 involves cancellation: tol reflects centering + SVD compound error
        np.testing.assert_allclose(float(rmsd), 0.0, atol=1e-6)

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
        # Scale = trace(D @ S) / var(P); well-conditioned SVD recovers to ~eps * cond
        np.testing.assert_allclose(float(c), c_true, rtol=1e-8, atol=1e-8)
        # RMSD ~ 0 involves cancellation: tol reflects centering + SVD compound error
        np.testing.assert_allclose(float(rmsd), 0.0, atol=1e-6)

    @_NUMPY_SETTINGS
    @given(nearly_collinear_3d())
    def test_rotation_is_not_unique_when_cross_covariance_is_degenerate(
        self, P_np: np.ndarray
    ) -> None:
        """Kabsch completes without error even when rotation is ambiguous.

        A near-collinear point cloud yields a near rank-1 cross-covariance H. The
        optimal rotation is not unique in this case: any rotation around the collinear
        axis achieves the same RMSD. This is the primary algorithm boundary users
        should be aware of -- SafeSVD produces a finite, stable (if arbitrary) rotation
        in this regime, but gradient-based optimizers should not rely on it to be the
        unique minimizer.
        """
        Q_np = P_np + np.random.default_rng(0).random(P_np.shape) * 0.5

        # kabsch must complete without raising and return valid outputs.
        R, t, rmsd = kabsch_np.kabsch(P_np, Q_np)
        assert np.isfinite(R).all()
        assert np.isfinite(t).all()
        assert np.isfinite(float(rmsd))
        # det(U @ V.T) is ±1 by construction; rounding in 3x3 matmul ~ D * eps
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

        # Perturb P along the dominant direction and verify kabsch still succeeds.
        direction = P_np[-1] - P_np[0]
        norm = np.linalg.norm(direction)
        assume(norm > 1e-10)
        direction = direction / norm
        P_perturbed = P_np + direction * 0.01

        R2, t2, rmsd2 = kabsch_np.kabsch(P_perturbed, Q_np)
        assert np.isfinite(R2).all()
        assert np.isfinite(t2).all()
        assert np.isfinite(float(rmsd2))
