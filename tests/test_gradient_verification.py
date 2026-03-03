import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        P_batch = adapter.convert_in(P_np)
        Q_batch = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        P_batch = adapter.convert_in(P_np)
        Q_batch = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

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
    def test_gradients_match_finite_differences_when_perturbed(
        self, adapter: FrameworkAdapter, algo: str, wrt: str, dim: int
    ) -> None:
        """
        Compares analytically computed gradients against numerical finite
        differences.
        """
        if dim >= 10:
            pytest.skip("Finite differences too slow for dim >= 10")
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        np.random.seed(42)
        n_points = max(10, dim * 2)
        P_np = np.random.rand(n_points, dim).astype(np.float64)
        Q_np = (P_np + np.random.rand(n_points, dim) * 0.1).astype(np.float64)
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch
        ref_adapter = type(adapter)("float64")
        func_ref = (
            ref_adapter.kabsch_umeyama if algo == "umeyama" else ref_adapter.kabsch
        )

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
    def test_gradients_match_finite_differences_when_purely_random(
        self, adapter: FrameworkAdapter, algo: str, wrt: str, dim: int
    ) -> None:
        """
        Compares analytically computed gradients against numerical finite
        differences for completely uncorrelated random point clouds.
        """
        if dim >= 10:
            pytest.skip("Finite differences too slow for dim >= 10")
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

        np.random.seed(123)
        n_points = max(10, dim * 2)
        P_np = np.random.rand(n_points, dim).astype(np.float64)
        Q_np = np.random.rand(n_points, dim).astype(np.float64)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch
        ref_adapter = type(adapter)("float64")
        func_ref = (
            ref_adapter.kabsch_umeyama if algo == "umeyama" else ref_adapter.kabsch
        )

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    def test_double_backward_is_computable_for_pytorch(
        self,
        algo: str,
    ) -> None:
        """
        Validates PyTorch implementation supports double backward (meta-learning).
        SVD double-backward frequently breaks due to mathematical singularities or
        framework limitations.
        """
        import torch

        from kabsch_horn import pytorch as kabsch_torch

        P = torch.rand((5, 3), dtype=torch.float64, requires_grad=True)
        Q = torch.rand((5, 3), dtype=torch.float64, requires_grad=True)

        func = kabsch_torch.kabsch_umeyama if algo == "umeyama" else kabsch_torch.kabsch

        res = func(P, Q)
        loss = sum([r.sum() for r in res])

        # First derivative
        grad_P = torch.autograd.grad(loss, P, create_graph=True)[0]

        # Second derivative check (sum of grad to condense to scalar)
        loss2 = grad_P.sum()
        try:
            loss2.backward()
        except RuntimeError as e:
            pytest.fail(f"Double backward failed: {e}")

        assert P.grad is not None
        assert torch.isfinite(P.grad).all()
