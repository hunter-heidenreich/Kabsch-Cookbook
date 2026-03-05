import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks


class TestErrorHandling:
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_raises_error_when_point_counts_differ(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Verifies that algorithms explicitly raise or propagate an error when
        P and Q have mismatched point counts.
        """
        dim = 3

        # 5 points vs 4 points
        P_np = np.random.rand(5, dim).astype(np.float64)
        Q_np = np.random.rand(4, dim).astype(np.float64)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

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
        dim = 3

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
        Verifies that if inputs contain NaNs, the output contains NaNs without
        raising hard C-level aborts or failing to track mathematically.
        """
        if adapter.__class__.__name__ == "MLXAdapter":
            pytest.skip(
                "MLX linalg.svd currently throws a fatal hardware Abort on NaNs "
                "which aborts pytest."
            )

        dim = 3

        import numpy as np

        np.random.seed(42)
        P_np = np.random.rand(5, dim).astype(np.float64)
        Q_np = np.random.rand(5, dim).astype(np.float64)

        # Inject NaN
        P_np[0, 0] = np.nan

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)

        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        try:
            res = func(P, Q)
        except Exception as e:
            pytest.skip(
                "Framework handles NaNs by raising an exception, "
                f"which is acceptable: {e}"
            )

        for tensor in res:
            if isinstance(tensor, float):
                import math

                assert math.isnan(tensor) or adapter.is_nan(tensor), (
                    "Expected NaN to propagate"
                )
            else:
                assert adapter.is_nan(tensor), "Expected NaN to propagate to output"
