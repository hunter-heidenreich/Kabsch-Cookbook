import math

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks


class TestErrorHandling:
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama", "horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_raises_error_when_point_counts_differ(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Verifies that all algorithms raise or propagate an error when
        P and Q have mismatched point counts (N).
        """
        rng = np.random.default_rng(0)
        # 5 points vs 4 points, 3D (horn requires 3D)
        P_np = rng.random((5, 3))
        Q_np = rng.random((4, 3))

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        with pytest.raises(adapter.mismatch_exception_type):
            func(P, Q)

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama", "horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_raises_error_when_dims_differ(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Verifies that all algorithms raise or propagate an error when
        P and Q have mismatched dimensionality (D).
        """
        rng = np.random.default_rng(0)
        P_np = rng.random((5, 3))
        Q_np = rng.random((5, 4))  # same N, different D

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        with pytest.raises(adapter.mismatch_exception_type):
            func(P, Q)

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_handles_underdetermined_systems_gracefully(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Verifies behavior when the system is underdetermined.
        The algorithms should mathematically succeed and find a perfect fit
        (RMSD approaches 0), even though the transform is not uniquely determined.
        """

        # 2 points in 3D (underdetermined)
        P_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        # Shift and rotate arbitrarily
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t_true = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        Q_np = (P_np @ R_true.T) + t_true

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)

        # The last return value is RMSD or a loss-like measure.
        rmsd_idx = -1
        rmsd = float(adapter.convert_out(res[rmsd_idx]))

        # An underdetermined system can always be fit perfectly.
        assert rmsd == pytest.approx(0.0, abs=adapter.atol)

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_propagates_nans_gracefully(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Contract: NaN inputs must propagate to NaN outputs without raising exceptions.

        PyTorch, JAX, and TensorFlow all propagate NaN through SVD. A framework
        that raises on NaN input would be a real test failure here.
        MLX is excluded because its linalg.svd fatally aborts the process on NaN.
        """
        if not adapter.supports_nan_input:
            pytest.skip(
                "MLX linalg.svd currently throws a fatal hardware Abort on NaNs "
                "which aborts pytest."
            )

        dim = 3

        rng = np.random.default_rng(42)
        P_np = rng.random((5, dim))
        Q_np = rng.random((5, dim))

        # Inject NaN
        P_np[0, 0] = np.nan

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)

        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)

        for tensor in res:
            if isinstance(tensor, float):
                assert math.isnan(tensor) or adapter.is_nan(tensor), (
                    "Expected NaN to propagate"
                )
            else:
                assert adapter.is_nan(tensor), "Expected NaN to propagate to output"
