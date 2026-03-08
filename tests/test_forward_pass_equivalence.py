from typing import TypeAlias

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
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


class TestForwardPassEquivalence:
    """
    Test suite verifying the forward pass equivalence of Kabsch and Umeyama algorithms.

    This test class validates the core functionality of the spatial transformation
    algorithms across different frameworks (NumPy, PyTorch, JAX, MLX, TensorFlow).

    It ensures:
    - Identity transformations are returned for identical point clouds.
    - Known geometric transformations (rotation, translation, scaling) are recovered
      correctly.
    - Framework-specific implementations exactly match a reference NumPy implementation.
    - N-dimensional batching logic computes exactly the same results as sequential
      processing.
    - Input data types (e.g., float32, float64) are strictly preserved in the outputs.
    """

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("kabsch", id="kabsch"),
            pytest.param("umeyama", id="umeyama"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_identity_transform_when_points_are_identical(
        self,
        identity_points: np.ndarray,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies that the algorithm returns identity transform for identical sets.

        Args:
            identity_points: The identity point cloud to test.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
        """
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

    @pytest.mark.parametrize(
        "algo, q_expected_idx, c_expected_source",
        [
            pytest.param("kabsch", 1, "constant", id="kabsch"),
            pytest.param("umeyama", 2, "c_true", id="umeyama"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_correct_transform_when_points_have_known_transformation(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
        q_expected_idx: int,
        c_expected_source: str,
    ) -> None:
        """Ensures that the algorithm computes expected matrix, translation, and scale.

        Args:
            known_transform_points: Tuple of P, Q_kabsch, Q_umeyama, R, t,
                c point clouds representing known ground truths.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
            q_expected_idx: The index in known_transform_points to extract Q.
            c_expected_source: Indicator for how to fetch c_expected.
        """
        P_np = known_transform_points[0]
        Q_expected = known_transform_points[q_expected_idx]
        R_true = known_transform_points[3]
        t_true = known_transform_points[4]
        c_true = known_transform_points[5]

        c_expected_map = {"constant": 1.0, "c_true": c_true}
        c_expected = c_expected_map[c_expected_source]

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
            None,
            algo,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )

    @pytest.mark.parametrize(
        "algo, q_expected_idx, ref_func",
        [
            pytest.param("kabsch", 1, _kabsch_numpy_adapter, id="kabsch"),
            pytest.param("umeyama", 2, _umeyama_numpy_adapter, id="umeyama"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_correct_transform_when_tested_against_numpy(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
        q_expected_idx: int,
        ref_func,
    ) -> None:
        """Validates that output of framework implementations matches reference NumPy.

        Args:
            known_transform_points: Point clouds representing known ground truths.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
            q_expected_idx: The index in known_transform_points to extract Q.
            ref_func: Reference function from numpy implementation.
        """
        P_np = known_transform_points[0]
        Q_expected = known_transform_points[q_expected_idx]

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)
        func = adapter.get_transform_func(algo)

        R_np, t_np, c_np, rmsd_np = ref_func(P_np, Q_expected)

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

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("kabsch", id="kabsch"),
            pytest.param("umeyama", id="umeyama"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_forward_pass_matches_sequential_computation_when_nd_batched(
        self,
        nd_batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies N-D batched forward passes match sequential computation outputs.

        Args:
            nd_batch_points: A tuple containing N-D batched P and Q arrays.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
        """
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

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("kabsch", id="kabsch"),
            pytest.param("umeyama", id="umeyama"),
        ],
    )
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

    @pytest.mark.parametrize(
        "algo, q_expected_idx",
        [
            pytest.param("kabsch", 1, id="kabsch"),
            pytest.param("umeyama", 2, id="umeyama"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_preserves_input_dtype_when_computing_transform(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
        q_expected_idx: int,
    ) -> None:
        """Verifies that the returned tensors maintain exactly the same dtype as inputs.

        Args:
            known_transform_points: Point clouds representing known ground truths.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
            q_expected_idx: The index in known_transform_points to extract Q.
        """
        P_np = known_transform_points[0]
        Q_expected = known_transform_points[q_expected_idx]

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


class TestHornForwardPassEquivalence:
    """
    Test suite verifying the forward pass equivalence of Horn and Horn-with-scale
    algorithms across all frameworks.

    Horn is 3D-only, so all fixtures are hardcoded to 3D and the `dim` fixture is
    never used here.
    """

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("horn", id="horn"),
            pytest.param("horn_with_scale", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_identity_when_points_identical(
        self,
        horn_identity_points: np.ndarray,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        P_np = horn_identity_points
        Q_np = np.copy(P_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)

        check_transform_close(
            adapter,
            res,
            np.eye(3),
            np.zeros(3),
            1.0,
            0.0,
            algo,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )

    @pytest.mark.parametrize(
        "algo, q_expected_idx, c_expected_source",
        [
            pytest.param("horn", 1, "constant", id="horn"),
            pytest.param("horn_with_scale", 2, "c_true", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_correct_transform_for_known_input(
        self,
        horn_known_transform_points: tuple,
        adapter: FrameworkAdapter,
        algo: str,
        q_expected_idx: int,
        c_expected_source: str,
    ) -> None:
        P_np = horn_known_transform_points[0]
        Q_np = horn_known_transform_points[q_expected_idx]
        R_true = horn_known_transform_points[3]
        t_true = horn_known_transform_points[4]
        c_true = horn_known_transform_points[5]

        c_expected = 1.0 if c_expected_source == "constant" else c_true

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P, Q)

        check_transform_close(
            adapter,
            res,
            R_true,
            t_true,
            c_expected,
            None,
            algo,
            atol=adapter.atol,
            rtol=adapter.rtol,
        )

    @pytest.mark.parametrize(
        "algo, q_expected_idx, ref_func",
        [
            pytest.param("horn", 1, _horn_numpy_adapter, id="horn"),
            pytest.param(
                "horn_with_scale",
                2,
                _horn_with_scale_numpy_adapter,
                id="horn_with_scale",
            ),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_matches_numpy_reference(
        self,
        horn_known_transform_points: tuple,
        adapter: FrameworkAdapter,
        algo: str,
        q_expected_idx: int,
        ref_func,
    ) -> None:
        P_np = horn_known_transform_points[0]
        Q_np = horn_known_transform_points[q_expected_idx]

        R_np, t_np, c_np, rmsd_np = ref_func(P_np, Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

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

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("horn", id="horn"),
            pytest.param("horn_with_scale", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_batched_matches_sequential(
        self,
        horn_nd_batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        P_np, Q_np = horn_nd_batch_points
        expected_tensors = compute_sequential_expected_tensors(
            P_np, Q_np, adapter, algo
        )

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        batch_res = func(P_fw, Q_fw)
        actual_tensors = [adapter.convert_out(t) for t in batch_res]

        for actual, expected in zip(actual_tensors, expected_tensors, strict=True):
            assert actual == pytest.approx(expected, rel=adapter.rtol, abs=adapter.atol)

    @pytest.mark.parametrize(
        "algo",
        [
            pytest.param("horn", id="horn"),
            pytest.param("horn_with_scale", id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_forward_pass_matches_sequential_computation_when_batched(
        self,
        horn_batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies [B, N, 3] batched Horn passes match sequential computation."""
        P_np, Q_np = horn_batch_points
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

    @pytest.mark.parametrize(
        "algo, q_expected_idx",
        [
            pytest.param("horn", 1, id="horn"),
            pytest.param("horn_with_scale", 2, id="horn_with_scale"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_preserves_dtype(
        self,
        horn_known_transform_points: tuple,
        adapter: FrameworkAdapter,
        algo: str,
        q_expected_idx: int,
    ) -> None:
        P_np = horn_known_transform_points[0]
        Q_np = horn_known_transform_points[q_expected_idx]

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
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
