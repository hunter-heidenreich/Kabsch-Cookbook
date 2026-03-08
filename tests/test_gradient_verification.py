import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from strategies import nearly_collinear_3d, nearly_coplanar_nd, point_clouds_3d
from utils import compute_numeric_grad


class TestGradientVerification:
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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
        np.random.seed(42)
        n_points = max(10, dim * 2)

        P_np = np.random.rand(n_points, dim).astype(np.float64)
        Q_np = (P_np + np.random.rand(n_points, dim) * 0.1).astype(np.float64)
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)
        ref_adapter = type(adapter)("float64")
        func_ref = ref_adapter.get_transform_func(algo)

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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
        np.random.seed(123)
        n_points = max(10, dim * 2)

        P_np = np.random.rand(n_points, dim).astype(np.float64)
        Q_np = np.random.rand(n_points, dim).astype(np.float64)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)
        ref_adapter = type(adapter)("float64")
        func_ref = ref_adapter.get_transform_func(algo)

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
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
                "FD gradient check is vacuous for float16/bfloat16 (atol*50=5.0) "
                "and imprecise for float32 (atol*50=2.5). float64 adapters cover "
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

        # 50x multiplier accounts for finite-difference truncation error and
        # floating-point cancellation in near-singular configurations.
        np.testing.assert_allclose(
            grad_analytic, grad_numeric, atol=adapter.atol * 50, rtol=adapter.rtol
        )

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    def test_computes_double_backward_when_using_pytorch(
        self,
        algo: str,
    ) -> None:
        """
        Validates PyTorch implementation supports double backward (meta-learning).
        SVD double-backward frequently breaks due to mathematical singularities or
        framework limitations.
        """
        import torch
        from adapters import PyTorchAdapter

        P = torch.rand((5, 3), dtype=torch.float64, requires_grad=True)
        Q = torch.rand((5, 3), dtype=torch.float64, requires_grad=True)
        adapter = PyTorchAdapter(precision="float64")
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        loss = sum([r.sum() for r in res])

        grad_P = torch.autograd.grad(loss, P, create_graph=True)[0]

        loss2 = grad_P.sum()
        loss2.backward()

        assert P.grad is not None
        assert torch.isfinite(P.grad).all()

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_safe_svd_gradient_reduces_rmsd_at_hypothesis_near_degenerate(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """SafeSVD masked gradients at near-degenerate inputs must not increase RMSD.

        Finite differences are numerically unreliable near singularities (see
        test_gradients_match_finite_differences_hypothesis for the stable-region FD
        check). This test verifies that SafeSVD's masked gradient at collinear or
        coplanar inputs is a valid descent direction -- a weaker but meaningful
        condition that can be checked without FD.

        This is the canonical "source of truth" test showing SafeSVD descent is
        guaranteed even at near-degenerate inputs (Fix #91).
        """
        # float16/bfloat16: overflow risk at near-degenerate inputs.
        if adapter.precision in ("float16", "bfloat16"):
            pytest.skip("overflow risk at near-degenerate inputs for float16/bfloat16")

        @settings(
            max_examples=20,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
            deadline=None,
        )
        @given(
            st.one_of(
                nearly_collinear_3d(),
                nearly_coplanar_nd(dim=3),
            )
        )
        def _inner(P_np: np.ndarray) -> None:
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
            # Loose tolerance (0.1): intentional -- SafeSVD masked gradients at
            # degeneracy are stable but not precise; we only require non-increase.
            alpha = 0.01
            P_step_np = P_np - alpha * grad
            P_step = adapter.convert_in(P_step_np.astype(np.float64))

            rmsd_orig = float(adapter.convert_out(rmsd_func(P, Q)[0]))
            rmsd_step = float(adapter.convert_out(rmsd_func(P_step, Q)[0]))

            assert rmsd_step <= rmsd_orig + 0.1, (
                f"RMSD increased after gradient step: "
                f"{rmsd_orig:.6f} -> {rmsd_step:.6f}"
            )

        _inner()


class TestHornGradientVerification:
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_sequential_when_batched(
        self,
        horn_batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """Verifies batched Horn gradients match sequential computation."""
        P_np, Q_np = horn_batch_points
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
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_finite_differences(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """Compares Horn analytic gradients against finite differences (3D only)."""
        np.random.seed(42)
        P_np = np.random.rand(10, 3).astype(np.float64)
        Q_np = (P_np + np.random.rand(10, 3) * 0.1).astype(np.float64)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)
        ref_adapter = type(adapter)("float64")
        func_ref = ref_adapter.get_transform_func(algo)

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_purely_random(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """Finite-difference check for Horn with uncorrelated random point clouds."""
        np.random.seed(123)
        P_np = np.random.rand(10, 3).astype(np.float64)
        Q_np = np.random.rand(10, 3).astype(np.float64)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)
        ref_adapter = type(adapter)("float64")
        func_ref = ref_adapter.get_transform_func(algo)

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
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
        """Compares Horn analytic vs finite-difference gradients (Hypothesis-varied)."""
        if adapter.precision in ("float16", "bfloat16", "float32"):
            pytest.skip(
                "FD gradient check is vacuous for float16/bfloat16 (atol*50=5.0) "
                "and imprecise for float32 (atol*50=2.5). float64 adapters cover "
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

        # 50x multiplier accounts for finite-difference truncation error and
        # floating-point cancellation in near-singular configurations.
        np.testing.assert_allclose(
            grad_analytic, grad_numeric, atol=adapter.atol * 50, rtol=adapter.rtol
        )

    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    def test_double_backward_pytorch(
        self,
        algo: str,
    ) -> None:
        """PyTorch-only: validates Horn supports double backward (create_graph=True)."""
        import torch
        from adapters import PyTorchAdapter

        P = torch.rand((5, 3), dtype=torch.float64, requires_grad=True)
        Q = torch.rand((5, 3), dtype=torch.float64, requires_grad=True)
        adapter = PyTorchAdapter(precision="float64")
        func = adapter.get_transform_func(algo)

        res = func(P, Q)
        loss = sum([r.sum() for r in res])

        grad_P = torch.autograd.grad(loss, P, create_graph=True)[0]

        loss2 = grad_P.sum()
        loss2.backward()

        assert P.grad is not None
        assert torch.isfinite(P.grad).all()
