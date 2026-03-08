import numpy as np
import pytest
from adapters import FrameworkAdapter, frameworks
from hypothesis import HealthCheck, assume, given, settings
from strategies import extreme_scale_cloud, nearly_collinear_3d, nearly_coplanar_nd


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

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(nearly_collinear_3d())
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_nearly_collinear_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
        P_np: np.ndarray,
    ) -> None:
        """Gradients remain finite for near-collinear point clouds (Hypothesis)."""
        assume(adapter.supports_dim(3))
        Q_np = P_np.copy()
        P = adapter.convert_in(P_np.astype(np.float64))
        Q = adapter.convert_in(Q_np.astype(np.float64))
        func = adapter.get_transform_func(algo)
        grad = adapter.get_grad(P, Q, func, wrt=wrt)
        assert np.all(np.isfinite(grad))

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(nearly_coplanar_nd())
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_nearly_coplanar_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
        P_np: np.ndarray,
    ) -> None:
        """Gradients remain finite for near-coplanar point clouds (Hypothesis)."""
        assume(adapter.supports_dim(P_np.shape[-1]))
        Q_np = P_np.copy()
        P = adapter.convert_in(P_np.astype(np.float64))
        Q = adapter.convert_in(Q_np.astype(np.float64))
        func = adapter.get_transform_func(algo)
        grad = adapter.get_grad(P, Q, func, wrt=wrt)
        assert np.all(np.isfinite(grad))

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(extreme_scale_cloud())
    @pytest.mark.parametrize("algo", ["kabsch", "umeyama"])
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_extreme_scale_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
        PQ: tuple,
    ) -> None:
        """Gradients remain finite for extreme-scale point clouds (Hypothesis)."""
        if getattr(adapter, "precision", "float64") not in ("float32", "float64"):
            pytest.skip("extreme scale unsafe for float16/bfloat16")
        P_np, Q_np = PQ
        assume(adapter.supports_dim(P_np.shape[-1]))
        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)
        grad = adapter.get_grad(P, Q, func, wrt=wrt)
        assert np.all(np.isfinite(grad))


class TestHornDifferentiabilityTraps:
    """
    Gradient stability tests for Horn quaternion algorithms (3D-only).

    Horn uses a different code path (quaternion eigensystem) than Kabsch (SVD),
    so degenerate inputs may trigger different numerical pitfalls. All fixtures
    are hardcoded to 3D.
    """

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_coplanar(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        rng = np.random.default_rng(42)
        P_np = rng.random((20, 3))
        P_np[:, -1] = 0.0
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        Q_np = P_np @ R.T + rng.random((3,))

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_collinear(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        rng = np.random.default_rng(42)
        P_np = np.zeros((20, 3))
        P_np[:, 0] = rng.random(20) * 10.0
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        Q_np = P_np @ R.T + rng.random((3,))

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_perfect_cube(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        points = []
        for i in range(3):
            p1, p2 = np.zeros(3), np.zeros(3)
            p1[i] = -1.0
            p2[i] = 1.0
            points.append(p1)
            points.append(p2)
        P_np = np.array(points, dtype=np.float64)
        Q_np = P_np + 0.5

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_returns_positive_det_for_reflected(
        self,
        adapter: FrameworkAdapter,
        algo: str,
    ) -> None:
        rng = np.random.default_rng(42)
        P_np = rng.random((20, 3))
        Q_np = np.copy(P_np)
        Q_np[:, 0] = -Q_np[:, 0]

        P_fw = adapter.convert_in(P_np)
        Q_fw = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        res = func(P_fw, Q_fw)
        R = adapter.convert_out(res[0])

        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=adapter.atol)

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_reflected(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        rng = np.random.default_rng(42)
        P_np = rng.random((20, 3))
        Q_np = np.copy(P_np)
        Q_np[:, 0] = -Q_np[:, 0]

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_identical(
        self,
        horn_identity_points: np.ndarray,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
    ) -> None:
        P_np = horn_identity_points
        Q_np = np.copy(P_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(nearly_collinear_3d())
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_nearly_collinear_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
        P_np: np.ndarray,
    ) -> None:
        """Horn gradients remain finite for near-collinear point clouds (Hypothesis)."""
        Q_np = P_np.copy()
        P = adapter.convert_in(P_np.astype(np.float64))
        Q = adapter.convert_in(Q_np.astype(np.float64))
        func = adapter.get_transform_func(algo)
        grad = adapter.get_grad(P, Q, func, wrt=wrt)
        assert np.all(np.isfinite(grad))

    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    @given(extreme_scale_cloud())
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_extreme_scale_hypothesis(
        self,
        adapter: FrameworkAdapter,
        wrt: str,
        algo: str,
        PQ: tuple,
    ) -> None:
        """Horn gradients remain finite for extreme-scale point clouds (Hypothesis)."""
        if getattr(adapter, "precision", "float64") not in ("float32", "float64"):
            pytest.skip("extreme scale unsafe for float16/bfloat16")
        P_np, Q_np = PQ
        P_np = P_np[:, :3] if P_np.shape[-1] > 3 else P_np
        Q_np = Q_np[:, :3] if Q_np.shape[-1] > 3 else Q_np
        # Pad to 3D if needed
        if P_np.shape[-1] < 3:
            pad = np.zeros((P_np.shape[0], 3 - P_np.shape[-1]))
            P_np = np.concatenate([P_np, pad], axis=-1)
            Q_np = np.concatenate([Q_np, pad], axis=-1)
        P = adapter.convert_in(P_np.astype(np.float64))
        Q = adapter.convert_in(Q_np.astype(np.float64))
        func = adapter.get_transform_func(algo)
        grad = adapter.get_grad(P, Q, func, wrt=wrt)
        assert np.all(np.isfinite(grad))

    @pytest.mark.parametrize("wrt", ["P", "Q"])
    @pytest.mark.parametrize("algo", ["horn", "horn_with_scale"])
    @pytest.mark.parametrize("collapse_target", ["P", "Q", "Both"])
    @pytest.mark.parametrize("adapter", frameworks)
    def test_gradients_stable_origin_collapse(
        self,
        adapter: FrameworkAdapter,
        algo: str,
        wrt: str,
        collapse_target: str,
    ) -> None:
        if algo == "horn_with_scale" and getattr(adapter, "precision", "float64") in (
            "float16",
            "bfloat16",
        ):
            pytest.skip(
                "Horn-with-scale variance division overflows float16"
                " on origin collapse."
            )

        np.random.seed(42)
        P_np = np.random.rand(5, 3).astype(np.float64)
        Q_np = np.random.rand(5, 3).astype(np.float64)

        if collapse_target in ["P", "Both"]:
            P_np = np.zeros_like(P_np)
        if collapse_target in ["Q", "Both"]:
            Q_np = np.zeros_like(Q_np)

        P = adapter.convert_in(P_np)
        Q = adapter.convert_in(Q_np)
        func = adapter.get_transform_func(algo)

        grad = adapter.get_grad(P, Q, func, wrt=wrt)

        assert np.isfinite(grad).all()
