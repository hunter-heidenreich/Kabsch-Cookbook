from typing import TypeAlias

import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from utils import check_transform_close

from kabsch_horn import numpy as kabsch_np

KnownTransformT: TypeAlias = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
]


def _compute_sequential_expected_tensors(
    P_np: np.ndarray,
    Q_np: np.ndarray,
    adapter: FrameworkAdapter,
    algo: str,
) -> list[np.ndarray]:
    """Computes expected batched outputs by running sequential computations.

    Args:
        P_np: Input N-D point cloud array.
        Q_np: Target N-D point cloud array.
        adapter: The framework adapter to use.
        algo: Algorithm to use ('kabsch' or 'umeyama').

    Returns:
        List of concatenated numpy arrays matching batched output structure.
    """
    func = adapter.get_transform_func(algo)
    b0, b1 = P_np.shape[0], P_np.shape[1]

    expected_res = []
    for i in range(b0):
        row_res = []
        for j in range(b1):
            P_seq = adapter.convert_in(P_np[i, j])
            Q_seq = adapter.convert_in(Q_np[i, j])
            seq_res = func(P_seq, Q_seq)
            row_res.append([adapter.convert_out(tensor) for tensor in seq_res])
        expected_res.append(row_res)

    num_tensors = len(expected_res[0][0])
    expected_tensors = []
    for t_idx in range(num_tensors):
        expected_tensor_list = [
            [expected_res[i][j][t_idx] for j in range(b1)] for i in range(b0)
        ]
        expected_tensors.append(np.array(expected_tensor_list))

    return expected_tensors


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
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        dim = P_np.shape[-1]

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
        "algo",
        [
            pytest.param("kabsch", id="kabsch"),
            pytest.param("umeyama", id="umeyama"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_correct_transform_when_points_have_known_transformation(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Ensures that the algorithm computes expected matrix, translation, and scale.

        Args:
            known_transform_points: Tuple of P, Q_kabsch, Q_umeyama, R, t,
                c point clouds representing known ground truths.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
        """
        P_np, Q_kabsch_np, Q_umeyama_np, R_true, t_true, c_true = known_transform_points
        Q_expected = Q_umeyama_np if algo == "umeyama" else Q_kabsch_np
        c_expected = c_true if algo == "umeyama" else 1.0
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
        "algo",
        [
            pytest.param("kabsch", id="kabsch"),
            pytest.param("umeyama", id="umeyama"),
        ],
    )
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_correct_transform_when_tested_against_numpy(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Validates that output of framework implementations matches reference NumPy.

        Args:
            known_transform_points: Point clouds representing known ground truths.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
        """
        P_np, Q_kabsch_np, Q_umeyama_np, _, _, _ = known_transform_points
        Q_expected = Q_umeyama_np if algo == "umeyama" else Q_kabsch_np
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)
        func = adapter.get_transform_func(algo)
        if algo == "umeyama":
            R_np, t_np, c_np, rmsd_np = kabsch_np.kabsch_umeyama(P_np, Q_expected)
        else:
            R_np, t_np, rmsd_np = kabsch_np.kabsch(P_np, Q_expected)
            c_np = 1.0

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
        expected_tensors = _compute_sequential_expected_tensors(
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
    def test_preserves_input_dtype_when_computing_transform(
        self,
        known_transform_points: KnownTransformT,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """Verifies that the returned tensors maintain exactly the same dtype as inputs.

        Args:
            known_transform_points: Point clouds representing known ground truths.
            adapter: Parameterized framework adapter.
            algo: The algorithm to test ('kabsch' or 'umeyama').
        """
        P_np, Q_kabsch_np, Q_umeyama_np, _, _, _ = known_transform_points
        Q_expected = Q_umeyama_np if algo == "umeyama" else Q_kabsch_np
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
