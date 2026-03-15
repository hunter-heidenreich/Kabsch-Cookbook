"""Tests for mixed-dtype input promotion (Issue #164).

When P and Q have different dtypes, both should be promoted to the
higher-precision type. Output dtype must match the promoted dtype.
"""

import numpy as np
import pytest
from adapters import frameworks
from conftest import ALGORITHMS
from scipy.spatial.transform import Rotation

# (P_precision, Q_precision, expected_output_precision)
DTYPE_PROMOTION_CASES = [
    ("float32", "float64", "float64"),
    ("float64", "float32", "float64"),
    ("float16", "float32", "float32"),
    ("float32", "float16", "float32"),
    ("float16", "float64", "float64"),
    ("bfloat16", "float32", "float32"),
    ("float16", "bfloat16", "float32"),
]


def _build_test_data(seed=42):
    """Build a fixed 3D test case with a known rotation."""
    rng = np.random.RandomState(seed)
    N = 10
    P_np = rng.randn(N, 3).astype(np.float64)
    R_true = Rotation.from_rotvec([0.3, 0.5, 0.1]).as_matrix()
    t_true = np.array([1.0, -2.0, 0.5])
    Q_np = (P_np @ R_true.T) + t_true
    return P_np, Q_np


class TestMixedDtypePromotion:
    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("p_prec,q_prec,expected_prec", DTYPE_PROMOTION_CASES)
    def test_mixed_dtype_output_precision(
        self, adapter, algo, p_prec, q_prec, expected_prec
    ):
        # Skip if the adapter doesn't support either precision
        if not hasattr(adapter, "_DTYPE_MAP"):
            pytest.skip("Adapter has no _DTYPE_MAP")
        if p_prec not in adapter._DTYPE_MAP or q_prec not in adapter._DTYPE_MAP:
            pytest.skip(f"Adapter lacks {p_prec} or {q_prec}")

        P_np, Q_np = _build_test_data()

        P = adapter.convert_in_with_dtype(P_np, p_prec)
        Q = adapter.convert_in_with_dtype(Q_np, q_prec)

        func = adapter.get_transform_func(algo)
        result = func(P, Q)

        # Check output dtype matches the promoted precision
        expected_dtype = adapter._DTYPE_MAP[expected_prec]
        for i, tensor in enumerate(result):
            out_dtype = tensor.dtype
            assert out_dtype == expected_dtype, (
                f"Output[{i}] dtype {out_dtype} != expected {expected_dtype} "
                f"for {adapter.name} {algo} with P={p_prec}, Q={q_prec}"
            )

    @pytest.mark.parametrize("adapter", frameworks)
    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("p_prec,q_prec,expected_prec", DTYPE_PROMOTION_CASES)
    def test_mixed_dtype_numerical_correctness(
        self, adapter, algo, p_prec, q_prec, expected_prec
    ):
        # Skip if the adapter doesn't support either precision
        if not hasattr(adapter, "_DTYPE_MAP"):
            pytest.skip("Adapter has no _DTYPE_MAP")
        if p_prec not in adapter._DTYPE_MAP or q_prec not in adapter._DTYPE_MAP:
            pytest.skip(f"Adapter lacks {p_prec} or {q_prec}")

        P_np, Q_np = _build_test_data()

        func = adapter.get_transform_func(algo)

        # Reference: quantize inputs through their respective low precisions,
        # then run same-dtype at the promoted precision. This isolates
        # promotion logic from quantization noise.
        P_quant = np.array(
            adapter.convert_out(adapter.convert_in_with_dtype(P_np, p_prec))
        )
        Q_quant = np.array(
            adapter.convert_out(adapter.convert_in_with_dtype(Q_np, q_prec))
        )
        P_ref = adapter.convert_in_with_dtype(P_quant, expected_prec)
        Q_ref = adapter.convert_in_with_dtype(Q_quant, expected_prec)
        ref_result = func(P_ref, Q_ref)

        # Mixed-dtype call
        P = adapter.convert_in_with_dtype(P_np, p_prec)
        Q = adapter.convert_in_with_dtype(Q_np, q_prec)
        mixed_result = func(P, Q)

        # Use tolerances based on the promoted precision
        tols = adapter._TOLERANCES[expected_prec]
        atol = tols["atol"]
        rtol = tols["rtol"]

        for i, (ref_t, mix_t) in enumerate(zip(ref_result, mixed_result, strict=True)):
            ref_arr = adapter.convert_out(ref_t)
            mix_arr = adapter.convert_out(mix_t)
            np.testing.assert_allclose(
                mix_arr,
                ref_arr,
                atol=atol,
                rtol=rtol,
                err_msg=(
                    f"Output[{i}] mismatch for {adapter.name} {algo} "
                    f"with P={p_prec}, Q={q_prec}"
                ),
            )
