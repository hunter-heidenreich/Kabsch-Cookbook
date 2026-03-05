import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks


class TestCatastrophicCancellation:
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_extreme_translation_preserves_rotation_and_translation(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Tests that scaling and large offsets (~1e6 or higher) don't cause
        catastrophic cancellation in the centroid calculations leading to
        wrong rotations and translations.
        """
        dim = 3

        if getattr(adapter, "precision", "float64") != "float64":
            pytest.skip(
                "Lower precisions inherently lose structure with extreme "
                "translations due to mantissa limits."
            )

        np.random.seed(42)
        P_np = np.random.rand(10, dim).astype(np.float64)

        # A large translation
        large_t = np.array([1e6, -2e6, 3e6], dtype=np.float64)

        # Applying a known rotation
        # 90 degrees around Z axis
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

        Q_np = (P_np @ R_true.T) + large_t

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)
        R_res = adapter.convert_out(res[0])
        t_res = adapter.convert_out(res[1])

        # High precision offset check because float32 suffers immense precision loss
        # at 1e6. If testing float32, tolerate higher absolute error.
        assert R_res == pytest.approx(R_true, abs=adapter.atol)

        # Translation error scales with magnitude due to float32 eps being a
        # relative concept. So we adjust the translation tolerance for the test
        assert t_res == pytest.approx(large_t, rel=adapter.rtol)

        if algo == "umeyama":
            c_res = float(adapter.convert_out(res[2]))
            assert c_res == pytest.approx(1.0, rel=adapter.rtol)

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_extreme_translation_of_both_point_clouds(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Tests when both P and Q are centered far from origin.
        """
        dim = 3

        if getattr(adapter, "precision", "float64") != "float64":
            pytest.skip(
                "Lower precisions inherently lose structure with extreme "
                "translations due to mantissa limits."
            )

        np.random.seed(42)
        P_np = np.random.rand(10, dim).astype(np.float64) * 10

        # extreme offsets
        offset_P = np.array([5e6, -4e6, 2e6], dtype=np.float64)
        large_t = np.array([100, -200, 300], dtype=np.float64)

        # Applying a known rotation
        # 90 degrees around Y axis
        R_true = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)

        P_shifted = P_np + offset_P
        Q_np = (P_np @ R_true.T) + offset_P + large_t

        P = adapter.convert_in(P_shifted)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)
        R_res = adapter.convert_out(res[0])
        t_res = adapter.convert_out(res[1])

        t_true = offset_P - (offset_P @ R_true.T) + large_t

        assert R_res == pytest.approx(R_true, abs=adapter.atol)
        assert t_res == pytest.approx(t_true, rel=adapter.rtol)

        if algo == "umeyama":
            c_res = float(adapter.convert_out(res[2]))
            assert c_res == pytest.approx(1.0, rel=adapter.rtol)
