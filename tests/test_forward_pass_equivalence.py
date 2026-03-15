from typing import TypeAlias

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from conftest import ALGORITHMS, ALGORITHMS_WITH_SCALE
from utils import check_transform_close, compute_sequential_expected_tensors

from kabsch_horn import numpy as kabsch_np

KnownTransformT: TypeAlias = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]


def _kabsch_numpy_adapter(P: np.ndarray, Q: np.ndarray):
    R, t, rmsd = kabsch_np.kabsch(P, Q)
    return R, t, 1.0, float(rmsd)


def _umeyama_numpy_adapter(P: np.ndarray, Q: np.ndarray):
    return kabsch_np.kabsch_umeyama(P, Q)


def _horn_numpy_adapter(P: np.ndarray, Q: np.ndarray):
    R, t, rmsd = kabsch_np.horn(P, Q)
    return R, t, 1.0, float(rmsd)


def _horn_with_scale_numpy_adapter(P: np.ndarray, Q: np.ndarray):
    return kabsch_np.horn_with_scale(P, Q)


# Maps algorithm name to the index in known_transform_points for Q
_Q_IDX = {"kabsch": 1, "umeyama": 2, "horn": 1, "horn_with_scale": 2}

# Maps algorithm name to its NumPy reference function
_REF_FUNCS = {
    "kabsch": _kabsch_numpy_adapter,
    "umeyama": _umeyama_numpy_adapter,
    "horn": _horn_numpy_adapter,
    "horn_with_scale": _horn_with_scale_numpy_adapter,
}


class TestForwardPassEquivalence:
    """
    Verifies forward pass correctness of Kabsch, Umeyama, Horn, and
    Horn-with-scale across all frameworks and dimensionalities.

    Horn tests are automatically skipped for dim != 3 via the collection hook.
    """

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_identity_transform_when_points_are_identical(
        self,
        identity_points: np.ndarray,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies that the algorithm returns identity transform for identical sets."""
        P_np = identity_points
        Q_np = np.copy(P_np)
        dim = P_np.shape[-1]

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)

        check_transform_close(
            adapter,
            res,
            np.eye(dim),
            np.zeros(dim),
            1.0,
            0.0,
            algo,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_correct_transform_when_points_have_known_transformation(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Ensures the algorithm computes expected rotation, translation, and scale."""
        P_np = known_transform_points[0]
        Q_expected = known_transform_points[_Q_IDX[algo]]
        R_true = known_transform_points[3]
        t_true = known_transform_points[4]
        c_true = known_transform_points[5]

        c_expected = c_true if algo in ALGORITHMS_WITH_SCALE else 1.0

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)

        check_transform_close(
            adapter,
            res,
            R_true,
            t_true,
            c_expected,
            0.0,
            algo,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_correct_transform_when_tested_against_numpy(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Validates that framework output matches reference NumPy implementation."""
        P_np = known_transform_points[0]
        Q_expected = known_transform_points[_Q_IDX[algo]]

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)
        func = adapter.get_transform_func(algo)

        R_np, t_np, c_np, rmsd_np = _REF_FUNCS[algo](P_np, Q_expected)

        res = func(P, Q)

        check_transform_close(
            adapter,
            res,
            R_np,
            t_np,
            c_np,
            rmsd_np,
            algo,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_forward_pass_matches_sequential_computation_when_nd_batched(
        self,
        nd_batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies N-D batched forward passes match sequential computation outputs."""
        P_np, Q_np = nd_batch_points
        expected_tensors = compute_sequential_expected_tensors(
            P_np, Q_np, adapter, algo
        )

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        batch_res = func(P_fw, Q_fw)

        actual_tensors = [adapter.convert_out(t) for t in batch_res]

        for actual, expected in zip(actual_tensors, expected_tensors, strict=True):
            assert actual == pytest.approx(
                expected,
                rel=adapter.rtol,
                abs=adapter.atol,
            )

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_forward_pass_matches_sequential_computation_when_batched(
        self,
        batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies [B, N, D] batched forward passes match sequential computation."""
        P_np, Q_np = batch_points
        func = adapter.get_transform_func(algo)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        batch_res = func(P_fw, Q_fw)
        batch_tensors = [adapter.convert_out(t) for t in batch_res]

        seq_res = [
            func(adapter.convert_in(P_np[i]), adapter.convert_in(Q_np[i]))
            for i in range(P_np.shape[0])
        ]
        for t_idx, actual in enumerate(batch_tensors):
            expected = np.stack(
                [adapter.convert_out(seq_res[i][t_idx]) for i in range(P_np.shape[0])]
            )
            assert actual == pytest.approx(expected, rel=adapter.rtol, abs=adapter.atol)

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_preserves_input_dtype_when_computing_transform(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies that returned tensors maintain the same dtype as inputs."""
        P_np = known_transform_points[0]
        Q_expected = known_transform_points[_Q_IDX[algo]]

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)

        func = adapter.get_transform_func(algo)
        expected_dtype = adapter._DTYPE_MAP[adapter.precision]

        res = func(P, Q)

        for tensor in res:
            assert hasattr(tensor, "dtype"), (
                f"Expected tensor to have 'dtype' attribute, got {type(tensor)}"
            )
            assert tensor.dtype == expected_dtype, (
                f"Expected dtype {expected_dtype}, got {tensor.dtype}"
            )

    @pytest.mark.parametrize("algo", ALGORITHMS)
    @pytest.mark.parametrize("adapter", frameworks)
    def test_reconstruction_aligns_point_clouds(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies that applying the recovered transform maps P onto Q."""
        P_np = known_transform_points[0]
        Q_np = known_transform_points[_Q_IDX[algo]]

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        res = adapter.get_transform_func(algo)(P, Q)

        R_out = adapter.convert_out(res[0])
        t_out = adapter.convert_out(res[1])
        c_out = (
            float(adapter.convert_out(res[2])) if algo in ALGORITHMS_WITH_SCALE else 1.0
        )

        P_aligned = c_out * (P_np @ R_out.T) + t_out
        assert P_aligned == pytest.approx(Q_np, rel=adapter.rtol, abs=adapter.atol)
