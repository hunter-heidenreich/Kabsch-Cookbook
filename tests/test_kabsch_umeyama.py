import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from utils import check_transform_close, compute_numeric_grad

from kabsch_umeyama import numpy as kabsch_np


class TestForwardPassEquivalence:
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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

        check_transform_close(
            adapter, res, np.eye(3), np.zeros(3), 1.0, 0.0, algo, atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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
            adapter, res, R_true, t_true, c_expected, None, algo, atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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
            adapter, res, R_np, t_np, c_np, rmsd_np, algo, atol=1e-4, rtol=1e-4
        )


class TestDifferentiabilityTraps:
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_coplanar(
        self,
        coplanar_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the input points
        are coplanar.
        """
        P_np, Q_np = coplanar_points
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_collinear(
        self,
        collinear_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the input points
        are collinear.
        """
        P_np, Q_np = collinear_points
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_form_perfect_cube(
        self,
        perfect_cube: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain stable when the input points form a perfect
        cube.
        """
        P_np, Q_np = perfect_cube
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_positive_determinant_when_points_are_reflected(
        self,
        reflected_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Ensures the calculated rotation matrix has a determinant of +1 for
        reflections.
        """
        P_np, Q_np = reflected_points
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P_fw, Q_fw)
        R = adapter.convert_out(res[0])

        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_reflected(
        self,
        reflected_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when inputs require
        reflection.
        """
        P_np, Q_np = reflected_points
        P_grad_in = adapter.convert_in(P_np)
        Q_grad_in = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad = adapter.get_grad(P_grad_in, Q_grad_in, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_are_identical(
        self,
        identity_points: np.ndarray,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the input points
        are identical (tests for RMSD gradient NaN trap).
        """
        P_np = identity_points
        Q_np = np.copy(P_np)
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()


class TestGradientVerification:
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_sequential_computation_when_batched(
        self,
        batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Verifies that batched gradients match sequential computation of those
        gradients.
        """
        P_np, Q_np = batch_points
        P_batch = adapter.convert_in(P_np)
        Q_batch = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad_batch = adapter.get_grad(P_batch, Q_batch, func, seed=None, wrt=wrt)

        grads_seq = []
        for i in range(5):
            P_seq = adapter.convert_in(P_np[i])
            Q_seq = adapter.convert_in(Q_np[i])
            g = adapter.get_grad(P_seq, Q_seq, func, seed=None, wrt=wrt)
            grads_seq.append(g)
        grad_seq_stacked = np.stack(grads_seq)

        assert grad_batch == pytest.approx(grad_seq_stacked, rel=1e-3, abs=1e-3)

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_finite_differences_when_perturbed(
        self, adapter: FrameworkAdapter, algo: str, wrt: str
    ) -> None:
        """
        Compares analytically computed gradients against numerical finite
        differences.
        """
        np.random.seed(42)
        P_np = np.random.rand(10, 3).astype(np.float64)
        Q_np = (P_np + np.random.rand(10, 3) * 0.1).astype(np.float64)
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(P_np, Q_np, adapter, func, wrt=wrt)

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )
