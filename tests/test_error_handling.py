import math

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from conftest import ALGORITHMS, ALGORITHMS_3D_ONLY


class TestErrorHandling:
    @pytest.mark.parametrize("algo", ALGORITHMS)
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

    @pytest.mark.parametrize("algo", ALGORITHMS)
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

    @pytest.mark.parametrize(
        "algo", [a for a in ALGORITHMS if a not in ALGORITHMS_3D_ONLY]
    )
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
        func = adapter.get_transform_func(algo)

        res = func(P, Q)

        # The last return value is RMSD or a loss-like measure.
        rmsd_idx = -1
        rmsd = float(adapter.convert_out(res[rmsd_idx]))

        # An underdetermined system can always be fit perfectly.
        assert rmsd == pytest.approx(0.0, abs=adapter.atol)

    @pytest.mark.parametrize(
        "algo", [a for a in ALGORITHMS if a not in ALGORITHMS_3D_ONLY]
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_propagates_nans_gracefully(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Contract: NaN inputs must propagate to NaN outputs without raising exceptions.

        PyTorch, JAX, TensorFlow, and MLX all propagate NaN through SVD/eigh.
        NumPy raises LinAlgError on NaN and is excluded via supports_nan_input.
        """
        if not adapter.supports_nan_input:
            pytest.skip(
                f"{adapter.name} does not support NaN inputs (raises or aborts)."
            )

        dim = 3

        rng = np.random.default_rng(42)
        P_np = rng.random((5, dim))
        Q_np = rng.random((5, dim))

        # Inject NaN
        P_np[0, 0] = np.nan

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)

        func = adapter.get_transform_func(algo)

        res = func(P, Q)

        for tensor in res:
            if isinstance(tensor, float):
                assert math.isnan(tensor) or adapter.is_nan(tensor), (
                    "Expected NaN to propagate"
                )
            else:
                assert adapter.is_nan(tensor), "Expected NaN to propagate to output"

    @pytest.mark.parametrize("algo", list(ALGORITHMS_3D_ONLY))
    @pytest.mark.parametrize("adapter", frameworks)
    def test_horn_propagates_nans_gracefully(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Horn algorithms should propagate NaN without crashing where supported.

        PyTorch and TensorFlow eigh raises on NaN-containing matrices, so only
        JAX and MLX (which have NaN guards) are expected to pass.
        """
        if not adapter.supports_nan_input:
            pytest.skip(
                f"{adapter.name} does not support NaN inputs (raises or aborts)."
            )

        rng = np.random.default_rng(42)
        P_np = rng.random((5, 3))
        Q_np = rng.random((5, 3))
        P_np[0, 0] = np.nan

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        try:
            res = func(P, Q)
        except Exception:
            # Some frameworks (PyTorch, TensorFlow) raise on NaN eigh input
            pytest.skip(f"{adapter.name} eigh raises on NaN input")

        for tensor in res:
            if isinstance(tensor, float):
                assert math.isnan(tensor) or adapter.is_nan(tensor), (
                    "Expected NaN to propagate"
                )
            else:
                assert adapter.is_nan(tensor), "Expected NaN to propagate to output"

    @pytest.mark.parametrize("algo", list(ALGORITHMS_3D_ONLY))
    @pytest.mark.parametrize(
        "dim", [pytest.param(2, id="2D"), pytest.param(4, id="4D")]
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_horn_raises_error_for_non_3d_input(self, adapter, algo, dim):
        """Horn algorithms must reject inputs where D != 3."""
        rng = np.random.default_rng(42)
        P = adapter.convert_in(rng.standard_normal((5, dim)).astype(np.float64))
        Q = adapter.convert_in(rng.standard_normal((5, dim)).astype(np.float64))
        func = adapter.get_transform_func(algo)
        with pytest.raises(ValueError, match="3D"):
            func(P, Q)


class TestRejects1DInput:
    """1D array inputs must raise ValueError, not IndexError."""

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_1d_input_raises_value_error(
        self, adapter: FrameworkAdapter, algo: str
    ) -> None:
        P_np = np.array([1.0, 2.0, 3.0])
        Q_np = np.array([4.0, 5.0, 6.0])

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        with pytest.raises(ValueError, match=r"at least 2D"):
            func(P, Q)


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


class TestMLXDeviceRestoration:
    """Verify _float64_device_guard restores the default device."""

    @pytest.mark.parametrize(
        "algo", ["kabsch", "kabsch_umeyama", "horn", "horn_with_scale"]
    )
    def test_device_restored_after_float64_call(self, algo: str) -> None:
        try:
            import mlx.core as mx

            from kabsch_horn import mlx as kabsch_mlx
        except ImportError:
            pytest.skip("MLX not available")

        # Set a known device before the call
        mx.set_default_device(mx.gpu)
        original = mx.default_device()

        rng = np.random.default_rng(0)
        P = mx.array(rng.random((5, 3)), dtype=mx.float64)
        Q = mx.array(rng.random((5, 3)), dtype=mx.float64)

        import warnings

        func_map = {
            "kabsch": kabsch_mlx.kabsch,
            "kabsch_umeyama": kabsch_mlx.kabsch_umeyama,
            "horn": kabsch_mlx.horn,
            "horn_with_scale": kabsch_mlx.horn_with_scale,
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            func_map[algo](P, Q)

        assert mx.default_device() == original, (
            f"Expected device {original} after float64 call, got {mx.default_device()}"
        )
