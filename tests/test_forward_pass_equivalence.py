import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from utils import check_transform_close

from kabsch_horn import numpy as kabsch_np


class TestForwardPassEquivalence:
    """
    Test suite verifying the forward pass equivalence of Kabsch and Umeyama algorithms.

    This test class validates the core functionality of the spatial transformation
    algorithms across different frameworks (NumPy, PyTorch, JAX, MLX, TensorFlow).
    
    It ensures:
    - Identity transformations are returned for identical point clouds.
    - Known geometric transformations (rotation, translation, scaling) are recovered correctly.
    - Framework-specific implementations exactly match a reference NumPy implementation.
    - N-dimensional batching logic computes exactly the same results as sequential processing.
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
        """
        Verifies that the algorithm returns an identity transform when both sets
        are identical.
        """
        P_np = identity_points
        Q_np = np.copy(P_np)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P, Q)

        dim = P_np.shape[-1]
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
        known_transform_points: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
        ],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Ensures that the algorithm computes expected matrix, translation, and
        scale for targets.
        """
        P_np, Q_kabsch_np, Q_umeyama_np, R_true, t_true, c_true = known_transform_points
        Q_expected = Q_umeyama_np if algo == "umeyama" else Q_kabsch_np
        c_expected = c_true if algo == "umeyama" else 1.0
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

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
        known_transform_points: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
        ],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Validates that output of framework implementations matches reference
        NumPy.
        """
        P_np, Q_kabsch_np, Q_umeyama_np, _, _, _ = known_transform_points
        Q_expected = Q_umeyama_np if algo == "umeyama" else Q_kabsch_np
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)

        if algo == "umeyama":
            func = adapter.kabsch_umeyama
            R_np, t_np, c_np, rmsd_np = kabsch_np.kabsch_umeyama(P_np, Q_expected)
        else:
            func = adapter.kabsch
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
        """
        Verifies that N-D batched forward passes match sequential computation of those
        forward passes.
        """
        P_np, Q_np = nd_batch_points

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch
        b0, b1 = P_np.shape[0], P_np.shape[1]

        batch_res = func(P_fw, Q_fw)
        seq_results = []
        for i in range(b0):
            row_res = []
            for j in range(b1):
                P_seq = adapter.convert_in(P_np[i, j])
                Q_seq = adapter.convert_in(Q_np[i, j])
                row_res.append(func(P_seq, Q_seq))
            seq_results.append(row_res)

        for i in range(b0):
            for j in range(b1):
                for b_tensor, s_tensor in zip(batch_res, seq_results[i][j], strict=False):
                    b_np = adapter.convert_out(b_tensor)
                    s_np = adapter.convert_out(s_tensor)
                    assert b_np[i, j] == pytest.approx(
                        s_np, rel=adapter.rtol, abs=adapter.atol
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
        known_transform_points: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
        ],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Verifies that the returned tensors maintain the exact same dtype as the inputs.
        """
        P_np, Q_kabsch_np, Q_umeyama_np, _, _, _ = known_transform_points
        Q_expected = Q_umeyama_np if algo == "umeyama" else Q_kabsch_np

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_expected)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch
        expected_dtype = adapter._DTYPE_MAP[adapter.precision]

        res = func(P, Q)

        for i, tensor in enumerate(res):
            if hasattr(tensor, "dtype"):
                assert tensor.dtype == expected_dtype, (
                    f"Failed dtype preservation on output index {i}. "
                    f"Expected {expected_dtype}, got {tensor.dtype}"
                )
