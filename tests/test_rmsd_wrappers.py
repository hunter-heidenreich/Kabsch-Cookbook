"""
Tests for kabsch_rmsd and kabsch_umeyama_rmsd single-call loss wrappers,
plus N=1 single-point edge cases.
"""

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks


class TestKabschRmsdWrappers:
    """Validates kabsch_rmsd and kabsch_umeyama_rmsd single-call loss functions."""

    @pytest.mark.parametrize("adapter", frameworks)
    def test_kabsch_rmsd_matches_kabsch_rmsd_output(
        self,
        adapter: FrameworkAdapter,
    ) -> None:
        """
        kabsch_rmsd(P, Q) must return the same RMSD as kabsch(P, Q)[2].
        """
        rng = np.random.default_rng(0)
        P_np = rng.random((20, 3))
        Q_np = rng.random((20, 3))

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)

        from kabsch_horn import jax as kabsch_jax
        from kabsch_horn import mlx as kabsch_mlx
        from kabsch_horn import pytorch as kabsch_torch
        from kabsch_horn import tensorflow as kabsch_tf

        adapter_cls = type(adapter).__name__
        if adapter_cls == "PyTorchAdapter":
            rmsd_wrapper = float(adapter.convert_out(kabsch_torch.kabsch_rmsd(P, Q)))
            rmsd_full = float(adapter.convert_out(kabsch_torch.kabsch(P, Q)[2]))
        elif adapter_cls == "JAXAdapter":
            rmsd_wrapper = float(adapter.convert_out(kabsch_jax.kabsch_rmsd(P, Q)))
            rmsd_full = float(adapter.convert_out(kabsch_jax.kabsch(P, Q)[2]))
        elif adapter_cls == "TFAdapter":
            rmsd_wrapper = float(adapter.convert_out(kabsch_tf.kabsch_rmsd(P, Q)))
            rmsd_full = float(adapter.convert_out(kabsch_tf.kabsch(P, Q)[2]))
        elif adapter_cls == "MLXAdapter":
            rmsd_wrapper = float(adapter.convert_out(kabsch_mlx.kabsch_rmsd(P, Q)))
            rmsd_full = float(adapter.convert_out(kabsch_mlx.kabsch(P, Q)[2]))
        else:
            pytest.skip(f"Unsupported adapter: {adapter_cls}")

        assert rmsd_wrapper == pytest.approx(
            rmsd_full, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("adapter", frameworks)
    def test_kabsch_umeyama_rmsd_matches_kabsch_umeyama_rmsd_output(
        self,
        adapter: FrameworkAdapter,
    ) -> None:
        """
        kabsch_umeyama_rmsd(P, Q) must return the same RMSD as kabsch_umeyama(P, Q)[3].
        """
        rng = np.random.default_rng(1)
        P_np = rng.random((20, 3))
        Q_np = rng.random((20, 3))

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)

        from kabsch_horn import jax as kabsch_jax
        from kabsch_horn import mlx as kabsch_mlx
        from kabsch_horn import pytorch as kabsch_torch
        from kabsch_horn import tensorflow as kabsch_tf

        adapter_cls = type(adapter).__name__
        if adapter_cls == "PyTorchAdapter":
            rmsd_w = float(adapter.convert_out(kabsch_torch.kabsch_umeyama_rmsd(P, Q)))
            rmsd_f = float(adapter.convert_out(kabsch_torch.kabsch_umeyama(P, Q)[3]))
        elif adapter_cls == "JAXAdapter":
            rmsd_w = float(adapter.convert_out(kabsch_jax.kabsch_umeyama_rmsd(P, Q)))
            rmsd_f = float(adapter.convert_out(kabsch_jax.kabsch_umeyama(P, Q)[3]))
        elif adapter_cls == "TFAdapter":
            rmsd_w = float(adapter.convert_out(kabsch_tf.kabsch_umeyama_rmsd(P, Q)))
            rmsd_f = float(adapter.convert_out(kabsch_tf.kabsch_umeyama(P, Q)[3]))
        elif adapter_cls == "MLXAdapter":
            rmsd_w = float(adapter.convert_out(kabsch_mlx.kabsch_umeyama_rmsd(P, Q)))
            rmsd_f = float(adapter.convert_out(kabsch_mlx.kabsch_umeyama(P, Q)[3]))
        else:
            pytest.skip(f"Unsupported adapter: {adapter_cls}")

        assert rmsd_w == pytest.approx(rmsd_f, rel=adapter.rtol, abs=adapter.atol)

    @pytest.mark.parametrize("algo", ["kabsch_rmsd", "kabsch_umeyama_rmsd"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_rmsd_wrapper_returns_zero_for_perfect_alignment(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        RMSD wrappers return ~0 when Q is a rigid transform of P.
        """
        rng = np.random.default_rng(2)
        n_points = 20
        P_np = rng.random((n_points, 3))

        # Random proper rotation
        A = rng.normal(size=(3, 3))
        Q_mat, R_mat = np.linalg.qr(A)
        d = np.diag(R_mat)
        R_true = Q_mat * (d / np.abs(d))
        if np.linalg.det(R_true) < 0:
            R_true[:, 0] *= -1

        t_true = rng.random((3,)) * 5.0 - 2.5
        Q_np = P_np @ R_true.T + t_true

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)

        from kabsch_horn import jax as kabsch_jax
        from kabsch_horn import mlx as kabsch_mlx
        from kabsch_horn import pytorch as kabsch_torch
        from kabsch_horn import tensorflow as kabsch_tf

        adapter_cls = type(adapter).__name__
        if adapter_cls == "PyTorchAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_torch.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_torch.kabsch_umeyama_rmsd,
            }
        elif adapter_cls == "JAXAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_jax.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_jax.kabsch_umeyama_rmsd,
            }
        elif adapter_cls == "TFAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_tf.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_tf.kabsch_umeyama_rmsd,
            }
        elif adapter_cls == "MLXAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_mlx.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_mlx.kabsch_umeyama_rmsd,
            }
        else:
            pytest.skip(f"Unsupported adapter: {adapter_cls}")

        rmsd = float(adapter.convert_out(func_map[algo](P, Q)))
        assert rmsd == pytest.approx(0.0, abs=adapter.atol)

    @pytest.mark.parametrize("algo", ["kabsch_rmsd", "kabsch_umeyama_rmsd"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_rmsd_wrapper_gradient_is_finite(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Gradients from rmsd wrappers are finite (no NaNs from the gradient-safe
        backward pass).
        """
        rng = np.random.default_rng(3)
        P_np = rng.random((20, 3))
        Q_np = rng.random((20, 3))

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)

        from kabsch_horn import jax as kabsch_jax
        from kabsch_horn import mlx as kabsch_mlx
        from kabsch_horn import pytorch as kabsch_torch
        from kabsch_horn import tensorflow as kabsch_tf

        adapter_cls = type(adapter).__name__
        if adapter_cls == "PyTorchAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_torch.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_torch.kabsch_umeyama_rmsd,
            }
        elif adapter_cls == "JAXAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_jax.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_jax.kabsch_umeyama_rmsd,
            }
        elif adapter_cls == "TFAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_tf.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_tf.kabsch_umeyama_rmsd,
            }
        elif adapter_cls == "MLXAdapter":
            func_map = {
                "kabsch_rmsd": kabsch_mlx.kabsch_rmsd,
                "kabsch_umeyama_rmsd": kabsch_mlx.kabsch_umeyama_rmsd,
            }
        else:
            pytest.skip(f"Unsupported adapter: {adapter_cls}")

        rmsd_fn = func_map[algo]

        def wrapper(P_inner, Q_inner):
            return (rmsd_fn(P_inner, Q_inner),)

        grad = adapter.get_grad(P, Q, wrapper, seed=None, wrt="P")
        assert np.isfinite(grad).all(), "Gradient from rmsd wrapper contains NaN/Inf"


class TestSinglePoint:
    """Edge case: N=1 (single point pair) -- underdetermined, but should not crash."""

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_single_point_returns_finite_outputs(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        A single point pair is maximally underdetermined. Both algorithms should
        succeed and return finite outputs (RMSD == 0 since perfect fit is trivial).
        """
        P_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        Q_np = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)

        for tensor in res:
            val = adapter.convert_out(tensor)
            assert np.isfinite(val).all(), f"Non-finite output in {algo} with N=1"

        rmsd = float(adapter.convert_out(res[-1]))
        assert rmsd == pytest.approx(0.0, abs=adapter.atol)
