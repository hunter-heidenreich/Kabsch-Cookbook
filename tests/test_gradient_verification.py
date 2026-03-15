import os

import numpy as np
import pytest
from adapters import (
    _JAX_AVAILABLE,
    _MLX_AVAILABLE,
    _TF_AVAILABLE,
    FrameworkAdapter,
    frameworks,
)
from conftest import ALGORITHMS
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from strategies import nearly_collinear_3d, nearly_coplanar_nd, point_clouds_3d
from utils import compute_numeric_grad

_FAST = os.environ.get("KABSCH_TEST_FAST") == "1"
_MAX_EXAMPLES_FD = 20 if _FAST else 100
_MAX_EXAMPLES_DEGEN = 15 if _FAST else 40


class TestGradientVerification:
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_sequential_computation_when_batched(
        self,
        batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Verifies that batched gradients match sequential computation of those
        gradients.
        """
        P_np, Q_np = batch_points
        P_batch = adapter.convert_in(P_np)
        Q_batch = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)

        grad_batch = adapter.get_grad(P_batch, Q_batch, func, seed=None, wrt=wrt)

        grads_seq = []
        for i in range(P_np.shape[0]):
            P_seq = adapter.convert_in(P_np[i])
            Q_seq = adapter.convert_in(Q_np[i])

            g = adapter.get_grad(P_seq, Q_seq, func, seed=None, wrt=wrt)
            grads_seq.append(g)

        grad_seq_stacked = np.stack(grads_seq)

        assert grad_batch == pytest.approx(
            grad_seq_stacked, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_sequential_computation_when_nd_batched(
        self,
        nd_batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Verifies that N-D batched gradients match sequential computation of those
        gradients.
        """
        P_np, Q_np = nd_batch_points
        P_batch = adapter.convert_in(P_np)
        Q_batch = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)

        grad_batch = adapter.get_grad(P_batch, Q_batch, func, seed=None, wrt=wrt)

        b0, b1 = P_np.shape[0], P_np.shape[1]

        grads_seq = np.zeros_like(P_np) if wrt == "P" else np.zeros_like(Q_np)
        for i in range(b0):
            for j in range(b1):
                P_seq = adapter.convert_in(P_np[i, j])
                Q_seq = adapter.convert_in(Q_np[i, j])

                g = adapter.get_grad(P_seq, Q_seq, func, seed=None, wrt=wrt)
                grads_seq[i, j] = g

        assert grad_batch == pytest.approx(
            grads_seq, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize(
        "dim",
        [
            pytest.param(2, id="2D"),
            pytest.param(3, id="3D"),
            pytest.param(4, id="4D"),
        ],
    )
    def test_gradients_match_finite_differences_when_perturbed(
        self, adapter: FrameworkAdapter, algo: str, wrt: str, dim: int
    ) -> None:
        """
        Compares analytically computed gradients against numerical finite
        differences.
        """
        rng = np.random.default_rng(42)
        n_points = max(10, dim * 2)

        P_np = rng.random((n_points, dim)).astype(np.float64)
        Q_np = (P_np + rng.random((n_points, dim)) * 0.1).astype(np.float64)
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)
        ref_adapter = type(adapter)("float64")
        func_ref = ref_adapter.get_transform_func(algo)

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        # No FD multiplier needed: well-conditioned inputs, float64
        # reference adapter, analytic grads in native precision.
        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize(
        "dim",
        [
            pytest.param(2, id="2D"),
            pytest.param(3, id="3D"),
            pytest.param(4, id="4D"),
        ],
    )
    def test_gradients_match_finite_differences_when_purely_random(
        self, adapter: FrameworkAdapter, algo: str, wrt: str, dim: int
    ) -> None:
        """
        Compares analytically computed gradients against numerical finite
        differences for completely uncorrelated random point clouds.
        """
        rng = np.random.default_rng(123)
        n_points = max(10, dim * 2)

        P_np = rng.random((n_points, dim)).astype(np.float64)
        Q_np = rng.random((n_points, dim)).astype(np.float64)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)
        ref_adapter = type(adapter)("float64")
        func_ref = ref_adapter.get_transform_func(algo)

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        # No FD multiplier needed: well-conditioned inputs, float64
        # reference adapter, analytic grads in native precision.
        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @settings(
        max_examples=_MAX_EXAMPLES_FD,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(point_clouds_3d(), st.integers(0, 2**31 - 1))
    def test_gradients_match_finite_differences_hypothesis(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
        P_np: np.ndarray,
        seed: int,
    ) -> None:
        """Compares analytic vs finite-difference gradients on Hypothesis inputs."""
        if adapter.precision in ("float16", "bfloat16", "float32"):
            pytest.skip(
                "FD gradient check is vacuous for float16/bfloat16 (atol*10=1.0) "
                "and borderline for float32 (atol*10=0.05, marginal given FD "
                "truncation error on random inputs). float64 adapters cover "
                "gradient correctness; deterministic FD tests cover float32 via "
                "float64 reference."
            )
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-1)
        rng = np.random.default_rng(seed)
        Q_np = (P_np + rng.standard_normal(P_np.shape)).astype(np.float64)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        ref_adapter = type(adapter)("float64")
        func_ref = ref_adapter.get_transform_func(algo)

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        # 10x multiplier accounts for finite-difference truncation error;
        # only float64 reaches here (float16/bfloat16/float32 are skipped above)
        # and near-singular inputs are rejected (sv[-1] > 0.1)
        np.testing.assert_allclose(
            grad_analytic, grad_numeric, atol=adapter.atol * 10, rtol=adapter.rtol
        )

    @pytest.mark.parametrize("precision", ["float32", "float64"])
    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_computes_double_backward_when_using_pytorch(
        self,
        algo: str,
        precision: str,
    ) -> None:
        """
        Validates PyTorch implementation supports double backward (meta-learning).
        SVD/eigh double-backward frequently breaks due to mathematical
        singularities or framework limitations.
        """
        import torch
        from adapters import PyTorchAdapter

        adapter = PyTorchAdapter(precision=precision)
        dtype = adapter._DTYPE_MAP[precision]
        P = torch.rand((5, 3), dtype=dtype, requires_grad=True)
        Q = torch.rand((5, 3), dtype=dtype, requires_grad=True)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        loss = sum([r.sum() for r in res])

        grad_P = torch.autograd.grad(loss, P, create_graph=True)[0]

        loss2 = grad_P.sum()
        loss2.backward()

        assert P.grad is not None
        assert torch.isfinite(P.grad).all()

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    @settings(
        max_examples=_MAX_EXAMPLES_DEGEN,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    @given(st.one_of(nearly_collinear_3d(), nearly_coplanar_nd(dim=3)))
    def test_safe_gradient_reduces_rmsd_at_hypothesis_near_degenerate(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        P_np: np.ndarray,
    ) -> None:
        """Safe masked gradients at near-degenerate inputs must not increase RMSD.

        Finite differences are numerically unreliable near singularities (see
        test_gradients_match_finite_differences_hypothesis for the stable-region FD
        check). This test verifies that masked gradients at collinear or coplanar
        inputs are valid descent directions -- a weaker but meaningful condition
        that can be checked without FD.
        """
        if adapter.precision in ("float16", "bfloat16"):
            pytest.skip("overflow risk at near-degenerate inputs for float16/bfloat16")

        # Skip inputs where the point cloud is so degenerate that the gradient
        # direction is unreliable (e.g., 4 nearly-identical points).
        sv = np.linalg.svd(P_np - P_np.mean(0), compute_uv=False)
        assume(sv[-1] > 1e-3)

        # Q is a small perturbation of P so RMSD > 0 but singularity is near.
        rng = np.random.default_rng(0)
        Q_np = (P_np + rng.standard_normal(P_np.shape) * 0.05).astype(np.float64)

        P = adapter.convert_in(P_np.astype(np.float64))
        Q = adapter.convert_in(Q_np.astype(np.float64))
        func = adapter.get_transform_func(algo)

        def rmsd_func(P_in, Q_in):
            return (func(P_in, Q_in)[-1],)

        grad = adapter.get_grad(P, Q, rmsd_func, seed=None, wrt="P")
        assert np.all(np.isfinite(grad)), (
            "gradient must be finite at near-degenerate inputs"
        )

        if np.linalg.norm(grad) < 1e-8:
            # gradient is effectively zero (at minimum or fully degenerate)
            return

        # Take one gradient step and verify RMSD does not increase.
        # Relative bound accommodates imprecision at degeneracy while
        # still catching genuinely bad (non-descent) gradients.
        alpha = 0.01
        P_step_np = P_np - alpha * grad
        P_step = adapter.convert_in(P_step_np.astype(np.float64))

        rmsd_orig = float(adapter.convert_out(rmsd_func(P, Q)[0]))
        rmsd_step = float(adapter.convert_out(rmsd_func(P_step, Q)[0]))

        # Relative bound: gradient step should not increase RMSD by more than 1%
        # plus eps for near-zero RMSD
        assert rmsd_step <= rmsd_orig * 1.01 + adapter.eps, (
            f"RMSD increased after gradient step: {rmsd_orig:.6f} -> {rmsd_step:.6f}"
        )


_JAX_SVD_XFAIL = pytest.mark.xfail(
    strict=True,
    reason=(
        "JAX custom_vjp does not implement SVD JVP; double backward through "
        "kabsch/kabsch_umeyama is unsupported upstream (jax.linalg.svd). "
        "Horn (eigh-based) is unaffected."
    ),
)

# Function names (not get_transform_func aliases) -- these tests call
# framework functions directly via algo_map lookups.
_DOUBLE_BACKWARD_ALGOS = ["kabsch", "kabsch_umeyama", "horn", "horn_with_scale"]
_PRECISIONS = ["float32", "float64"]


class TestDoubleBackwardNonPyTorch:
    """Double backward coverage for JAX, TensorFlow, and MLX.

    PyTorch double backward is tested in TestGradientVerification. This class
    extends that coverage to the remaining autodiff frameworks.

    JAX kabsch/kabsch_umeyama are marked xfail(strict=True): JAX's custom_vjp
    does not implement an SVD JVP, so double backward raises NotImplementedError
    upstream. Horn algorithms use eigh and are unaffected.
    """

    @pytest.mark.skipif(not _JAX_AVAILABLE, reason="JAX not installed")
    @pytest.mark.parametrize("precision", _PRECISIONS)
    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("kabsch", marks=_JAX_SVD_XFAIL),
            pytest.param("kabsch_umeyama", marks=_JAX_SVD_XFAIL),
            "horn",
            "horn_with_scale",
        ],
    )
    def test_double_backward_jax(self, algo: str, precision: str) -> None:
        """JAX double backward via jax.grad applied twice."""
        import jax
        import jax.numpy as jnp

        from kabsch_horn import jax as kh

        algo_map = {
            "kabsch": kh.kabsch,
            "kabsch_umeyama": kh.kabsch_umeyama,
            "horn": kh.horn,
            "horn_with_scale": kh.horn_with_scale,
        }
        dtype = jnp.float32 if precision == "float32" else jnp.float64

        rng = np.random.default_rng(42)
        P = jnp.array(rng.random((10, 3)), dtype=dtype)
        Q = jnp.array(rng.random((10, 3)), dtype=dtype)
        func = algo_map[algo]

        def loss_fn(P_in: jax.Array) -> jax.Array:
            return sum(jnp.sum(r) for r in func(P_in, Q))

        grad2_fn = jax.grad(lambda P_in: jnp.sum(jax.grad(loss_fn)(P_in)))
        g2 = grad2_fn(P)

        assert jnp.all(jnp.isfinite(g2))

    @pytest.mark.skipif(not _TF_AVAILABLE, reason="TensorFlow not installed")
    @pytest.mark.parametrize("precision", _PRECISIONS)
    @pytest.mark.parametrize("algo", _DOUBLE_BACKWARD_ALGOS)
    def test_double_backward_tensorflow(self, algo: str, precision: str) -> None:
        """TensorFlow double backward via nested GradientTape."""
        import tensorflow as tf

        from kabsch_horn import tensorflow as kh

        algo_map = {
            "kabsch": kh.kabsch,
            "kabsch_umeyama": kh.kabsch_umeyama,
            "horn": kh.horn,
            "horn_with_scale": kh.horn_with_scale,
        }
        dtype = tf.float32 if precision == "float32" else tf.float64

        rng = np.random.default_rng(42)
        P = tf.Variable(rng.random((10, 3)), dtype=dtype)
        Q = tf.Variable(rng.random((10, 3)), dtype=dtype)
        func = algo_map[algo]

        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                res = func(P, Q)
                loss = sum(tf.reduce_sum(r) for r in res)
            grad = tape1.gradient(loss, P)
            grad_sum = tf.reduce_sum(grad)
        grad2 = tape2.gradient(grad_sum, P)

        assert grad2 is not None
        assert tf.reduce_all(tf.math.is_finite(grad2))

    @pytest.mark.skipif(not _MLX_AVAILABLE, reason="MLX not installed")
    @pytest.mark.parametrize("precision", _PRECISIONS)
    @pytest.mark.parametrize("algo", _DOUBLE_BACKWARD_ALGOS)
    def test_double_backward_mlx(self, algo: str, precision: str) -> None:
        """MLX double backward via mx.grad applied twice."""
        import mlx.core as mx

        from kabsch_horn import mlx as kh

        algo_map = {
            "kabsch": kh.kabsch,
            "kabsch_umeyama": kh.kabsch_umeyama,
            "horn": kh.horn,
            "horn_with_scale": kh.horn_with_scale,
        }
        dtype = mx.float32 if precision == "float32" else mx.float64
        mx.set_default_device(mx.cpu if precision == "float64" else mx.gpu)

        rng = np.random.default_rng(42)
        P = mx.array(rng.random((10, 3)), dtype=dtype)
        Q = mx.array(rng.random((10, 3)), dtype=dtype)
        func = algo_map[algo]

        def loss_fn(P_in: mx.array) -> mx.array:
            return sum(mx.sum(r) for r in func(P_in, Q))

        grad2_fn = mx.grad(lambda P_in: mx.sum(mx.grad(loss_fn)(P_in)))
        g2 = grad2_fn(P)
        mx.eval(g2)

        assert mx.all(mx.isfinite(g2)).item()
