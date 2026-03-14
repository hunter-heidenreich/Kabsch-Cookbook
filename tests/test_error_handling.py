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

        with pytest.raises(adapter.mismatch_exception_type, match=r"same shape"):
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

        with pytest.raises(adapter.mismatch_exception_type, match=r"same shape"):
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


class TestNumpySinglePointRejection:
    """NumPy N=1 inputs must raise ValueError for all algorithms."""

    @pytest.mark.parametrize(
        "algo", ["kabsch", "kabsch_umeyama", "horn", "horn_with_scale"]
    )
    def test_numpy_single_point_raises_value_error(self, algo: str) -> None:
        from kabsch_horn.numpy import horn, horn_with_scale, kabsch, kabsch_umeyama

        P = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        Q = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)

        func_map = {
            "kabsch": kabsch,
            "kabsch_umeyama": kabsch_umeyama,
            "horn": horn,
            "horn_with_scale": horn_with_scale,
        }

        with pytest.raises(ValueError, match="2 points"):
            func_map[algo](P, Q)


class TestNumpyFloat16Upcast:
    """NumPy float16 inputs should be upcast internally (not raise TypeError)."""

    @pytest.mark.parametrize(
        "algo", ["kabsch", "kabsch_umeyama", "horn", "horn_with_scale"]
    )
    def test_numpy_float16_no_type_error(self, algo: str) -> None:
        from kabsch_horn.numpy import horn, horn_with_scale, kabsch, kabsch_umeyama

        rng = np.random.default_rng(0)
        P = rng.random((5, 3)).astype(np.float16)
        Q = rng.random((5, 3)).astype(np.float16)

        func_map = {
            "kabsch": kabsch,
            "kabsch_umeyama": kabsch_umeyama,
            "horn": horn,
            "horn_with_scale": horn_with_scale,
        }
        result = func_map[algo](P, Q)

        for arr in result:
            assert arr.dtype == np.float16, f"Expected float16 output, got {arr.dtype}"

    @pytest.mark.parametrize(
        "algo", ["kabsch", "kabsch_umeyama", "horn", "horn_with_scale"]
    )
    def test_numpy_float16_batched(self, algo: str) -> None:
        from kabsch_horn.numpy import horn, horn_with_scale, kabsch, kabsch_umeyama

        rng = np.random.default_rng(1)
        P = rng.random((2, 5, 3)).astype(np.float16)
        Q = rng.random((2, 5, 3)).astype(np.float16)

        func_map = {
            "kabsch": kabsch,
            "kabsch_umeyama": kabsch_umeyama,
            "horn": horn,
            "horn_with_scale": horn_with_scale,
        }
        result = func_map[algo](P, Q)

        for arr in result:
            assert arr.dtype == np.float16, f"Expected float16 output, got {arr.dtype}"

    @pytest.mark.parametrize("algo", ["kabsch", "kabsch_umeyama"])
    def test_numpy_float16_higher_dim(self, algo: str) -> None:
        from kabsch_horn.numpy import kabsch, kabsch_umeyama

        rng = np.random.default_rng(2)
        P = rng.random((5, 4)).astype(np.float16)
        Q = rng.random((5, 4)).astype(np.float16)

        func = kabsch if algo == "kabsch" else kabsch_umeyama
        result = func(P, Q)

        for arr in result:
            assert arr.dtype == np.float16, f"Expected float16 output, got {arr.dtype}"


class TestNumpyFloat32DtypePromotion:
    """NumPy functions should preserve float32 (not promote to float64)."""

    @pytest.mark.parametrize(
        "algo", ["kabsch", "kabsch_umeyama", "horn", "horn_with_scale"]
    )
    def test_float32_stays_float32(self, algo: str) -> None:
        from kabsch_horn.numpy import horn, horn_with_scale, kabsch, kabsch_umeyama

        rng = np.random.default_rng(0)
        P = rng.random((5, 3)).astype(np.float32)
        Q = rng.random((5, 3)).astype(np.float32)

        func_map = {
            "kabsch": kabsch,
            "kabsch_umeyama": kabsch_umeyama,
            "horn": horn,
            "horn_with_scale": horn_with_scale,
        }
        result = func_map[algo](P, Q)

        for arr in result:
            assert arr.dtype == np.float32, f"Expected float32 output, got {arr.dtype}"
