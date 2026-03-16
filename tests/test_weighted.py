"""Tests for per-point weighted alignment across all frameworks and algorithms."""

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from conftest import ALGORITHMS, ALGORITHMS_3D_ONLY, ALGORITHMS_WITH_SCALE
from utils import compute_numeric_grad


def _get_transform_func(adapter, algo):
    return adapter.get_transform_func(algo)


def _call_algo(adapter, algo, P, Q, weights=None):
    func = _get_transform_func(adapter, algo)
    return func(P, Q, weights=weights)


def _get_rmsd(res, algo):
    if algo in ALGORITHMS_WITH_SCALE:
        return res[3]
    return res[2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[2, 3, 4], ids=lambda x: f"{x}D")
def dim(request) -> int:
    return request.param


@pytest.fixture
def points_and_transform(dim):
    """Generate source points P and target Q = R @ P + t with known transform."""
    rng = np.random.default_rng(42)
    n_points = max(10, dim * 2)
    P = rng.random((n_points, dim))

    # Random rotation (Haar measure via QR)
    A = rng.normal(size=(dim, dim))
    Q_mat, R_mat = np.linalg.qr(A)
    d = np.diag(R_mat)
    ph = d / np.abs(d)
    rot = Q_mat * ph
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1

    t_true = rng.random((dim,)) * 5.0 - 2.5
    Q = P @ rot.T + t_true
    return P, Q, rot, t_true


# ---------------------------------------------------------------------------
# Test: uniform weights == no weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", frameworks, indirect=False)
@pytest.mark.parametrize("algo", ALGORITHMS)
class TestUniformWeightsEquivalence:
    def test_uniform_ones(
        self, adapter: FrameworkAdapter, algo, dim, points_and_transform
    ):
        """weights=ones(N) should produce identical results to weights=None."""
        if not adapter.supports_dim(dim):
            pytest.skip("Adapter does not support this dim")
        if algo in ALGORITHMS_3D_ONLY and dim != 3:
            pytest.skip("3D-only algorithm")

        P_np, Q_np, _, _ = points_and_transform
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        w_np = np.ones(P_np.shape[0], dtype=np.float64)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        res_no_w = _call_algo(adapter, algo, P_fw, Q_fw, weights=None)
        res_w = _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)

        for t_no_w, t_w in zip(res_no_w, res_w, strict=False):
            np.testing.assert_allclose(
                adapter.convert_out(t_w),
                adapter.convert_out(t_no_w),
                atol=adapter.atol,
                rtol=adapter.rtol,
            )

    def test_uniform_scaled(
        self, adapter: FrameworkAdapter, algo, dim, points_and_transform
    ):
        """weights=k*ones(N) for arbitrary k>0 should match weights=None."""
        if not adapter.supports_dim(dim):
            pytest.skip("Adapter does not support this dim")
        if algo in ALGORITHMS_3D_ONLY and dim != 3:
            pytest.skip("3D-only algorithm")

        P_np, Q_np, _, _ = points_and_transform
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        w_np = np.full(P_np.shape[0], 7.3, dtype=np.float64)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        res_no_w = _call_algo(adapter, algo, P_fw, Q_fw, weights=None)
        res_w = _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)

        for t_no_w, t_w in zip(res_no_w, res_w, strict=False):
            np.testing.assert_allclose(
                adapter.convert_out(t_w),
                adapter.convert_out(t_no_w),
                atol=adapter.atol,
                rtol=adapter.rtol,
            )


# ---------------------------------------------------------------------------
# Test: weight scaling invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", frameworks, indirect=False)
@pytest.mark.parametrize("algo", ALGORITHMS)
class TestWeightScalingInvariance:
    def test_scaling_invariance(
        self, adapter: FrameworkAdapter, algo, dim, points_and_transform
    ):
        """weights=w and weights=k*w should produce the same R, t, c."""
        if not adapter.supports_dim(dim):
            pytest.skip("Adapter does not support this dim")
        if algo in ALGORITHMS_3D_ONLY and dim != 3:
            pytest.skip("3D-only algorithm")

        P_np, Q_np, _, _ = points_and_transform
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        rng = np.random.default_rng(123)
        w_np = rng.random(P_np.shape[0]).astype(np.float64) + 0.1
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        w_scaled_np = w_np * 5.5
        w_scaled_fw = adapter.convert_in_with_dtype(w_scaled_np, adapter.precision)

        res_w = _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)
        res_ws = _call_algo(adapter, algo, P_fw, Q_fw, weights=w_scaled_fw)

        for t_w, t_ws in zip(res_w, res_ws, strict=False):
            np.testing.assert_allclose(
                adapter.convert_out(t_ws),
                adapter.convert_out(t_w),
                atol=adapter.atol,
                rtol=adapter.rtol,
            )


# ---------------------------------------------------------------------------
# Test: zero-weight outlier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", frameworks, indirect=False)
@pytest.mark.parametrize("algo", ALGORITHMS)
class TestZeroWeightOutlier:
    def test_zero_weight_outlier(
        self, adapter: FrameworkAdapter, algo, dim, points_and_transform
    ):
        """Setting weight=0 on an outlier should recover the clean transform."""
        if not adapter.supports_dim(dim):
            pytest.skip("Adapter does not support this dim")
        if algo in ALGORITHMS_3D_ONLY and dim != 3:
            pytest.skip("3D-only algorithm")

        P_np, Q_np, R_true, t_true = points_and_transform
        n = P_np.shape[0]

        # Add an outlier as the last point
        outlier_P = np.array([100.0] * dim, dtype=np.float64)
        outlier_Q = np.array([-100.0] * dim, dtype=np.float64)
        P_with_outlier = np.vstack([P_np, outlier_P[np.newaxis, :]])
        Q_with_outlier = np.vstack([Q_np, outlier_Q[np.newaxis, :]])

        # Zero weight on the outlier
        w_np = np.ones(n + 1, dtype=np.float64)
        w_np[-1] = 0.0

        P_fw = adapter.convert_in(P_with_outlier)
        Q_fw = adapter.convert_in(Q_with_outlier)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        res = _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)

        R_out = adapter.convert_out(res[0])
        t_out = adapter.convert_out(res[1])

        np.testing.assert_allclose(
            R_out, R_true, atol=adapter.atol * 10, rtol=adapter.rtol * 10
        )
        np.testing.assert_allclose(
            t_out, t_true, atol=adapter.atol * 10, rtol=adapter.rtol * 10
        )


# ---------------------------------------------------------------------------
# Test: weighted RMSD manual check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", frameworks, indirect=False)
@pytest.mark.parametrize("algo", ALGORITHMS)
class TestWeightedRMSD:
    def test_weighted_rmsd(
        self, adapter: FrameworkAdapter, algo, dim, points_and_transform
    ):
        """Weighted RMSD should match manual computation."""
        if not adapter.supports_dim(dim):
            pytest.skip("Adapter does not support this dim")
        if algo in ALGORITHMS_3D_ONLY and dim != 3:
            pytest.skip("3D-only algorithm")

        P_np, Q_np, _, _ = points_and_transform
        rng = np.random.default_rng(99)
        w_np = rng.random(P_np.shape[0]).astype(np.float64) + 0.1

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        res = _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)
        R_out = adapter.convert_out(res[0])
        t_out = adapter.convert_out(res[1])
        rmsd_out = float(adapter.convert_out(_get_rmsd(res, algo)))

        if algo in ALGORITHMS_WITH_SCALE:
            c_out = float(adapter.convert_out(res[2]))
            aligned = c_out * (P_np @ R_out.T) + t_out
        else:
            aligned = P_np @ R_out.T + t_out

        residual_sq = np.sum(np.square(aligned - Q_np), axis=-1)
        w_sum = np.sum(w_np)
        rmsd_manual = np.sqrt(np.sum(w_np * residual_sq) / w_sum)

        np.testing.assert_allclose(
            rmsd_out, rmsd_manual, atol=adapter.atol * 5, rtol=adapter.rtol * 5
        )


# ---------------------------------------------------------------------------
# Test: validation errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", frameworks, indirect=False)
@pytest.mark.parametrize("algo", ALGORITHMS)
class TestWeightValidation:
    def test_negative_weight(self, adapter: FrameworkAdapter, algo):
        """Negative weights should raise for all algorithms."""
        if not adapter.supports_dim(3):
            pytest.skip("Adapter does not support dim=3")

        rng = np.random.default_rng(42)
        P_np = rng.random((5, 3))
        Q_np = rng.random((5, 3))
        w_np = np.array([1.0, -1.0, 1.0, 1.0, 1.0])

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        with pytest.raises((ValueError, Exception)):
            _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)

    def test_all_zero_weights(self, adapter: FrameworkAdapter, algo):
        """All-zero weights should raise for all algorithms."""
        if not adapter.supports_dim(3):
            pytest.skip("Adapter does not support dim=3")

        rng = np.random.default_rng(42)
        P_np = rng.random((5, 3))
        Q_np = rng.random((5, 3))
        w_np = np.zeros(5)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        with pytest.raises((ValueError, Exception)):
            _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)

    def test_wrong_shape_weight(self, adapter: FrameworkAdapter, algo):
        """Wrong-shaped weights should raise for all algorithms."""
        if not adapter.supports_dim(3):
            pytest.skip("Adapter does not support dim=3")

        rng = np.random.default_rng(42)
        P_np = rng.random((5, 3))
        Q_np = rng.random((5, 3))
        w_np = np.ones(4)  # Wrong length

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        with pytest.raises((ValueError, Exception)):
            _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)


# ---------------------------------------------------------------------------
# Test: batched weights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", frameworks, indirect=False)
@pytest.mark.parametrize("algo", ALGORITHMS)
class TestBatchedWeights:
    def test_batched_weights(self, adapter: FrameworkAdapter, algo):
        """Batched weights [B, N] with [B, N, D] point clouds."""
        dim = 3
        if not adapter.supports_dim(dim):
            pytest.skip("Adapter does not support dim=3")
        if algo in ALGORITHMS_3D_ONLY and dim != 3:
            pytest.skip("3D-only algorithm")

        rng = np.random.default_rng(42)
        B, N = 3, 8
        P_np = rng.random((B, N, dim))
        Q_np = rng.random((B, N, dim))
        w_np = rng.random((B, N)).astype(np.float64) + 0.1

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        res = _call_algo(adapter, algo, P_fw, Q_fw, weights=w_fw)

        # Check output shapes
        R_out = adapter.convert_out(res[0])
        assert R_out.shape == (B, dim, dim)

        # Compare with sequential per-batch computation
        func = _get_transform_func(adapter, algo)
        for b in range(B):
            P_b = adapter.convert_in(P_np[b])
            Q_b = adapter.convert_in(Q_np[b])
            w_b = adapter.convert_in_with_dtype(w_np[b], adapter.precision)
            res_b = func(P_b, Q_b, weights=w_b)

            R_b = adapter.convert_out(res_b[0])
            R_batch_b = adapter.convert_out(res[0])[b]
            np.testing.assert_allclose(
                R_batch_b, R_b, atol=adapter.atol, rtol=adapter.rtol
            )


# ---------------------------------------------------------------------------
# Test: gradient correctness for weighted alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", frameworks, indirect=False)
@pytest.mark.parametrize("algo", ALGORITHMS)
class TestWeightedGradient:
    def test_weighted_gradient(self, adapter: FrameworkAdapter, algo):
        """Analytical gradient with weights matches finite differences."""
        if not adapter.supports_grad:
            pytest.skip("Forward-only adapter")
        dim = 3
        if not adapter.supports_dim(dim):
            pytest.skip("Adapter does not support dim=3")
        if algo in ALGORITHMS_3D_ONLY and dim != 3:
            pytest.skip("3D-only algorithm")
        if adapter.precision in ("float16", "bfloat16"):
            pytest.skip("Finite-difference gradient check unreliable at low precision")

        rng = np.random.default_rng(42)
        N = 8
        P_np = rng.random((N, dim)).astype(np.float64)
        Q_np = rng.random((N, dim)).astype(np.float64)
        w_np = rng.random(N).astype(np.float64) + 0.1

        w_fw = adapter.convert_in_with_dtype(w_np, adapter.precision)

        # Create a weighted function for gradient computation
        func_name = algo
        base_func = adapter.get_transform_func(func_name)

        def weighted_func(P, Q):
            return base_func(P, Q, weights=w_fw)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        grad_analytical = adapter.get_grad(P_fw, Q_fw, weighted_func, seed=42, wrt="P")

        grad_numeric = compute_numeric_grad(
            P_np, Q_np, adapter, weighted_func, seed=42, wrt="P"
        )

        np.testing.assert_allclose(
            grad_analytical,
            grad_numeric,
            atol=adapter.atol * 10,
            rtol=adapter.rtol * 10,
        )
