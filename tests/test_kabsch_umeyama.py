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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
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

        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
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

        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
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

    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        batch_res = func(P_fw, Q_fw)

        b0, b1 = P_np.shape[0], P_np.shape[1]
        for i in range(b0):
            for j in range(b1):
                P_seq = adapter.convert_in(P_np[i, j])
                Q_seq = adapter.convert_in(Q_np[i, j])
                seq_res = func(P_seq, Q_seq)

                for b_tensor, s_tensor in zip(batch_res, seq_res, strict=False):
                    b_np = adapter.convert_out(b_tensor)
                    s_np = adapter.convert_out(s_tensor)
                    assert b_np[i, j] == pytest.approx(
                        s_np, rel=adapter.rtol, abs=adapter.atol
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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        res = func(P_fw, Q_fw)
        R = adapter.convert_out(res[0])

        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=adapter.atol)

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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_system_is_underdetermined(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the system is
        underdetermined.
        """
        dim = 3
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

        # 2 points in 3D (underdetermined)
        P_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t_true = np.array([2.0, 3.0, 4.0], dtype=np.float64)
        Q_np = (P_np @ R_true.T) + t_true

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("collapse_target", ["P", "Q", "Both"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_are_stable_when_points_collapse_to_origin(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
        collapse_target: str,
    ) -> None:
        """
        Checks that gradients remain numerically stable when the inputs collapse
        to the origin.
        """
        dim = 3
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

        if algo == "umeyama" and getattr(adapter, "precision", "float64") in (
            "float16",
            "bfloat16",
        ):
            pytest.skip(
                "Umeyama requires division by variance, which overflows float16 on "
                "origin collapse."
            )

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

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize(
        "fixture_name",
        [
            pytest.param("coplanar_points", id="coplanar"),
            pytest.param("collinear_points", id="collinear"),
            pytest.param("perfect_cube", id="perfect_cube"),
            pytest.param("reflected_points", id="reflected"),
            pytest.param("identity_points", id="identity"),
        ],
    )
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_point_in_descent_direction_at_singularities(
        self,
        request,
        fixture_name: str,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
        dim: int,
    ) -> None:
        """
        Validates that the computed gradients at singularities actually point in a
        direction that decreases the RMSD.
        """
        points = request.getfixturevalue(fixture_name)
        if isinstance(points, tuple):
            P_np, Q_np = points
        else:
            P_np = points
            Q_np = np.copy(P_np)

        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        def rmsd_func(P_inner, Q_inner):
            res = func(P_inner, Q_inner)
            return (res[-1],)

        grad = adapter.get_grad(P, Q, rmsd_func, seed=None, wrt=wrt)

        if np.linalg.norm(grad) < adapter.atol:
            assert np.isfinite(grad).all()
            return

        alpha = max(1e-4, adapter.eps)
        if wrt == "P":
            P_new_np = P_np - alpha * grad
            P_new = adapter.convert_in(P_new_np)
            Q_new = Q
        else:
            Q_new_np = Q_np - alpha * grad
            Q_new = adapter.convert_in(Q_new_np)
            P_new = P

        rmsd_orig = float(adapter.convert_out(rmsd_func(P, Q)[0]))
        rmsd_new = float(adapter.convert_out(rmsd_func(P_new, Q_new)[0]))

        # With lower precision, floating point noise around zero can mask the descent.
        assert rmsd_new < rmsd_orig + adapter.eps


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
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        P_batch = adapter.convert_in(P_np)
        Q_batch = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad_batch = adapter.get_grad(P_batch, Q_batch, func, seed=None, wrt=wrt)

        grads_seq = []
        for i in range(P_np.shape[0]):
            P_seq = adapter.convert_in(P_np[i])
            Q_seq = adapter.convert_in(Q_np[i])
            g = adapter.get_grad(P_seq, Q_seq, func, seed=None, wrt=wrt)
            grads_seq.append(g)
        grad_seq_stacked = np.stack(grads_seq)

        assert grad_batch == pytest.approx(
            grad_seq_stacked, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_sequential_computation_when_nd_batched(
        self,
        nd_batch_points: tuple[np.ndarray, np.ndarray],
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        """
        Verifies that N-D batched gradients match sequential computation of those
        gradients.
        """
        P_np, Q_np = nd_batch_points
        dim = P_np.shape[-1]
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        P_batch = adapter.convert_in(P_np)
        Q_batch = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        grad_batch = adapter.get_grad(P_batch, Q_batch, func, seed=None, wrt=wrt)

        b0, b1 = P_np.shape[0], P_np.shape[1]
        grads_seq = np.zeros_like(P_np) if wrt == "P" else np.zeros_like(Q_np)
        for i in range(b0):
            for j in range(b1):
                P_seq = adapter.convert_in(P_np[i, j])
                Q_seq = adapter.convert_in(Q_np[i, j])
                g = adapter.get_grad(P_seq, Q_seq, func, seed=None, wrt=wrt)
                grads_seq[i, j] = g

        assert grad_batch == pytest.approx(
            grads_seq, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_finite_differences_when_perturbed(
        self, adapter: FrameworkAdapter, algo: str, wrt: str, dim: int
    ) -> None:
        """
        Compares analytically computed gradients against numerical finite
        differences.
        """
        if dim >= 10:
            pytest.skip("Finite differences too slow for dim >= 10")
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")
        np.random.seed(42)
        n_points = max(10, dim * 2)
        P_np = np.random.rand(n_points, dim).astype(np.float64)
        Q_np = (P_np + np.random.rand(n_points, dim) * 0.1).astype(np.float64)
        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch
        ref_adapter = type(adapter)("float64")
        func_ref = (
            ref_adapter.kabsch_umeyama if algo == "umeyama" else ref_adapter.kabsch
        )

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_match_finite_differences_when_purely_random(
        self, adapter: FrameworkAdapter, algo: str, wrt: str, dim: int
    ) -> None:
        """
        Compares analytically computed gradients against numerical finite
        differences for completely uncorrelated random point clouds.
        """
        if dim >= 10:
            pytest.skip("Finite differences too slow for dim >= 10")
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

        np.random.seed(123)
        n_points = max(10, dim * 2)
        P_np = np.random.rand(n_points, dim).astype(np.float64)
        Q_np = np.random.rand(n_points, dim).astype(np.float64)

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch
        ref_adapter = type(adapter)("float64")
        func_ref = (
            ref_adapter.kabsch_umeyama if algo == "umeyama" else ref_adapter.kabsch
        )

        grad_analytic = adapter.get_grad(P_fw, Q_fw, func, wrt=wrt)
        grad_numeric = compute_numeric_grad(
            P_np, Q_np, ref_adapter, func_ref, wrt=wrt, weight_adapter=adapter
        )

        assert grad_analytic == pytest.approx(
            grad_numeric, rel=adapter.rtol, abs=adapter.atol
        )


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
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

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
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

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


class TestErrorHandling:
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_raises_error_when_point_counts_differ(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        """
        Verifies that algorithms explicitly raise or propagate an error when
        P and Q have mismatched point counts.
        """
        dim = 3
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

        # 5 points vs 4 points
        P_np = np.random.rand(5, dim).astype(np.float64)
        Q_np = np.random.rand(4, dim).astype(np.float64)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.kabsch_umeyama if algo == "umeyama" else adapter.kabsch

        with pytest.raises(adapter.mismatch_exception_type):
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
        dim = 3
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

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
        if not adapter.supports_dim(dim):
            pytest.skip(f"{adapter.__class__.__name__} doesn't support {dim}D")

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
