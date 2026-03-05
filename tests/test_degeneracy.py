import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks


class TestDegeneracy:
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("collapse_target", ["P", "Q", "Both"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_origin_collapse_returns_identity_rotation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        collapse_target: str,
    ) -> None:
        """
        Verifies behavior when points collapse to the origin.
        Scale should theoretically go zero (for umeyama).
        """
        dim = 3

        np.random.seed(42)
        P_np = np.random.rand(5, dim).astype(np.float64)
        Q_np = np.random.rand(5, dim).astype(np.float64)

        if collapse_target in ["P", "Both"]:
            P_np = np.zeros_like(P_np)
        if collapse_target in ["Q", "Both"]:
            Q_np = np.zeros_like(Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)
        R_res = adapter.convert_out(res[0])

        # Rotation should be identity to remain well-conditioned
        assert R_res == pytest.approx(np.eye(dim), abs=adapter.atol)

        # Scale check for Umeyama
        if algo == "umeyama":
            c_res = float(adapter.convert_out(res[2]))
            if collapse_target in ["P", "Q", "Both"]:
                assert np.isfinite(c_res)
