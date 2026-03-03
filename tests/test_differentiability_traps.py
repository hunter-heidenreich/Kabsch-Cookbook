import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks


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
